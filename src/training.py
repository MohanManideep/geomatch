"""Reusable optimizer, EMA, schedule, fingerprint and checkpoint machinery."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_fingerprint(fold: int) -> dict:
    paths = {
        "architecture_config": ROOT / "configs/architecture.json",
        "data_config": ROOT / "configs/data.json",
        "geography_config": ROOT / "configs/geography.json",
        "objective_config": ROOT / "configs/objective.json",
        "training_config": ROOT / "configs/training.json",
        "official_csv": ROOT / "splits/official_folds_seed42.csv",
        "normalization": ROOT / f"artifacts/normalization/fold_{fold}.json",
        "geography_report": ROOT / "artifacts/official_geography_locked/report.json",
        "training_assignments": ROOT
        / f"artifacts/official_geography_locked/fold_{fold}/training_assignments.csv",
        "validation_assignments": ROOT
        / f"artifacts/official_geography_locked/fold_{fold}/validation_assignments.csv",
        "prototypes": ROOT
        / f"artifacts/official_geography_locked/fold_{fold}/combination_prototypes.csv",
        "model_source": ROOT / "src/model.py",
        "data_source": ROOT / "src/data.py",
        "decoder_source": ROOT / "src/decoder.py",
        "objective_source": ROOT / "src/objective.py",
        "scoring_source": ROOT / "src/scoring.py",
        "training_source": ROOT / "src/training.py",
        "train_entrypoint": ROOT / "src/07_train.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Fingerprint inputs missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    combined = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"fold": fold, "files": hashes, "combined_sha256": combined}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie in (0, 1)")
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.target_decay = float(decay)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.target_decay, 1.0 - 1.0 / (self.updates + 1.0))
        source_parameters = dict(model.named_parameters())
        for name, ema_parameter in self.model.named_parameters():
            ema_parameter.mul_(decay).add_(
                source_parameters[name].detach(), alpha=1.0 - decay
            )
        source_buffers = dict(model.named_buffers())
        for name, ema_buffer in self.model.named_buffers():
            ema_buffer.copy_(source_buffers[name])

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "updates": self.updates,
            "target_decay": self.target_decay,
        }

    def load_state_dict(self, state: dict) -> None:
        if abs(float(state["target_decay"]) - self.target_decay) > 1e-12:
            raise ValueError("EMA decay differs from checkpoint")
        self.model.load_state_dict(state["model"], strict=True)
        self.updates = int(state["updates"])


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    backbone_names = {
        name for name, _ in model.stem.named_parameters(prefix="stem")
    } | {name for name, _ in model.stages.named_parameters(prefix="stages")}
    groups: dict[str, dict] = {}
    specifications = {
        "backbone_decay": (
            float(optimizer_config["backbone_learning_rate"]),
            float(optimizer_config["backbone_weight_decay"]),
        ),
        "backbone_no_decay": (float(optimizer_config["backbone_learning_rate"]), 0.0),
        "head_decay": (
            float(optimizer_config["head_learning_rate"]),
            float(optimizer_config["head_weight_decay"]),
        ),
        "head_no_decay": (float(optimizer_config["head_learning_rate"]), 0.0),
    }
    for group_name, (learning_rate, weight_decay) in specifications.items():
        groups[group_name] = {
            "name": group_name,
            "params": [],
            "lr": learning_rate,
            "initial_lr": learning_rate,
            "weight_decay": weight_decay,
        }
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name in backbone_names
        use_decay = parameter.ndim >= 2
        group_name = ("backbone" if is_backbone else "head") + (
            "_decay" if use_decay else "_no_decay"
        )
        groups[group_name]["params"].append(parameter)
        if id(parameter) in seen:
            raise RuntimeError(f"Parameter appears twice: {name}")
        seen.add(id(parameter))
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if seen != expected:
        raise RuntimeError(
            "Optimizer groups do not cover every trainable parameter exactly once"
        )
    return torch.optim.AdamW(
        list(groups.values()),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["epsilon"]),
    )


class WarmupCosineSchedule:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        minimum_factor: float,
    ) -> None:
        if not 0 <= warmup_steps < total_steps:
            raise ValueError("Warmup steps must be in [0, total_steps)")
        if not 0.0 < minimum_factor <= 1.0:
            raise ValueError("minimum_factor must be in (0, 1]")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.minimum_factor = float(minimum_factor)
        self.initial_lrs = [
            float(group["initial_lr"]) for group in optimizer.param_groups
        ]

    def factor(self, step: int) -> float:
        if self.warmup_steps and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps - 1
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_factor + (1.0 - self.minimum_factor) * cosine

    def apply(self, step: int) -> list[float]:
        factor = self.factor(step)
        values = []
        for group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            group["lr"] = initial_lr * factor
            values.append(group["lr"])
        return values


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
