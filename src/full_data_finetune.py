"""Retrain the final recipe on all 11,758 labelled images (no held-out fold)
to produce the single model that is submitted.

Differences from the cross-validation driver (``country_aware_retrieval_finetune``):

* the training set is every labelled image, so there is no validation fold and
  no per-epoch retrieval evaluation -- the run is a fixed 40-epoch schedule;
* the trunk is initialised from an existing BYOL backbone (unsupervised, no
  labels), and the frozen teacher descriptors for the listwise distillation
  term are the per-fold encoder cache concatenated to cover all images.

Objective, sampler, hard-negative mining, anchor oversampling and the optimiser
schedule are imported unchanged from the cross-validation driver.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import country_aware_retrieval_finetune as cv
from data import DEFAULT_DATA_CONFIG, GeoMatchFoldDataset, ThreeViewTransform, read_json
from model import count_trainable_parameters
from retrieval_model import EXPECTED_RETRIEVAL_PARAMETER_COUNT, GeoCPRegNetRetrieval
from retrieval_objective import build_spatial_objective
from training import (
    ModelEMA,
    WarmupCosineSchedule,
    atomic_json,
    atomic_torch_save,
    build_optimizer,
    set_seed,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
TRAINER = cv.TRAINER
COMMON = cv.COMMON

DEFAULT_IMAGES = Path("/var/tmp/luli38se-geomatch/data/geo_dataset/train")
DEFAULT_EVIDENCE = Path(
    "/var/tmp/luli38se-geomatch/outputs/regnet_cp_v2_failure_audit/evidence/fold_0.npz"
)
DEFAULT_BACKBONE = Path(
    "/var/tmp/luli38se-geomatch/outputs/regnet_cp_ssl_backbone_full/fold_2/backbone.pt"
)
DEFAULT_OUTPUT = Path("/var/tmp/luli38se-geomatch/outputs/regnet_cp_full_data")
DEFAULT_NORMALIZATION = ROOT / "artifacts/normalization/fold_0.json"
ASSET_ROOT = ROOT / "artifacts/fold_assignments"
CHECKPOINT_FORMAT = "geomatch-regnet-full-data-retrieval-checkpoint"
BACKBONE_FORMAT = "geomatch-regnet-cp-ssl-byol-backbone-v1"


def load_all_rows() -> pd.DataFrame:
    """Every labelled image, in a fixed filename-sorted order."""
    frames = []
    for fold in range(5):
        frames.append(
            pd.read_csv(ASSET_ROOT / f"fold_{fold}" / "validation_assignments.csv")
        )
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.drop_duplicates("filename").sort_values("filename")
    rows = rows.reset_index(drop=True)
    if len(rows) != 11_758:
        raise RuntimeError(f"expected 11,758 images, assembled {len(rows)}")
    return rows


def load_teacher_descriptors(evidence_path: Path, rows: pd.DataFrame) -> np.ndarray:
    """Frozen fused encoder descriptors for every image, in ``rows`` order."""
    with np.load(evidence_path, allow_pickle=False) as source:
        names = np.concatenate(
            [source["train_filename"].astype(str), source["val_filename"].astype(str)]
        )
        fused = np.concatenate(
            [
                source["train_descriptor_fused"].astype(np.float32),
                source["val_descriptor_fused"].astype(np.float32),
            ]
        )
    lookup = {name: i for i, name in enumerate(names)}
    missing = [n for n in rows["filename"] if n not in lookup]
    if missing:
        raise RuntimeError(f"teacher descriptors missing {len(missing)} images")
    order = np.array([lookup[n] for n in rows["filename"]], dtype=np.int64)
    return fused[order]


def load_backbone(
    path: Path, device: torch.device
) -> tuple[GeoCPRegNetRetrieval, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != BACKBONE_FORMAT:
        raise ValueError(f"Unexpected SSL backbone format at {path}")
    model = GeoCPRegNetRetrieval(local_features=64, local_grid_size=4).to(device)
    backbone_state = checkpoint["backbone"]
    incompatible = model.load_state_dict(backbone_state, strict=False)
    model_keys = set(model.state_dict())
    if not set(backbone_state).issubset(model_keys):
        raise RuntimeError("SSL backbone has keys absent from the model")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected SSL keys: {incompatible.unexpected_keys[:5]}")
    expected_missing = {
        k for k in model_keys if not (k.startswith("stem.") or k.startswith("stages."))
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError("SSL load touched more than the trunk")
    if count_trainable_parameters(model) != EXPECTED_RETRIEVAL_PARAMETER_COUNT:
        raise RuntimeError("parameter count differs from the locked retrieval count")
    provenance = {
        "backbone": str(path),
        "sha256": sha256_file(path),
        "ssl_epochs": int(checkpoint.get("epoch", -1)),
        "ssl_final_loss": checkpoint.get("final_loss"),
    }
    return model, provenance


def build_datasets(rows: pd.DataFrame, images: Path):
    config = read_json(DEFAULT_DATA_CONFIG)
    normalization = read_json(DEFAULT_NORMALIZATION)
    train_transform = ThreeViewTransform(
        normalization["mean"], normalization["std"], training=True, config=config
    )
    bank_transform = ThreeViewTransform(
        normalization["mean"], normalization["std"], training=False, config=config
    )
    return (
        GeoMatchFoldDataset(rows, images, train_transform),
        GeoMatchFoldDataset(rows, images, bank_transform),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=cv.DEFAULT_CONFIG)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-batches-limit", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("format") != cv.CONFIG_FORMAT:
        raise ValueError("Unexpected configuration format")
    seed = int(config["seed"])
    set_seed(seed)
    torch.backends.cudnn.benchmark = True

    rows = load_all_rows()
    filenames = rows["filename"].astype(str).to_numpy()
    coordinates = rows[["lat", "lng"]].to_numpy(dtype=np.float64)
    countries = rows["country_index"].to_numpy(dtype=np.int64)
    cached_teacher = load_teacher_descriptors(args.evidence, rows).astype(np.float32)
    filename_to_position = {name: i for i, name in enumerate(filenames)}

    train_dataset, bank_dataset = build_datasets(rows, args.images.resolve())
    model, start_provenance = load_backbone(args.backbone, device)
    ema = ModelEMA(model, decay=float(config["ema"]["decay"]))
    spatial_objective = build_spatial_objective(
        config["objective"]["spatial_retrieval"]
    ).to(device)
    classification = cv.JointGeoClassificationObjective(
        label_smoothing=float(config["objective"]["classification"]["label_smoothing"]),
        weights={
            n: float(v)
            for n, v in config["objective"]["classification"]["terms"].items()
        },
    ).to(device)
    optimizer = build_optimizer(model, config)

    physical_batch = int(config["batch"]["physical_batch"])
    accumulation = int(config["batch"]["gradient_accumulation_steps"])
    workers = int(config["batch"]["workers"])
    batches_per_epoch = len(train_dataset) // physical_batch
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    maximum_epochs = int(config["checkpoint"]["fixed_final_epoch"])
    stop_after = (
        maximum_epochs
        if args.stop_after_epoch is None
        else min(maximum_epochs, int(args.stop_after_epoch))
    )
    schedule = WarmupCosineSchedule(
        optimizer,
        total_steps=steps_per_epoch * maximum_epochs,
        warmup_steps=steps_per_epoch * int(config["schedule"]["warmup_epochs"]),
        minimum_factor=float(config["schedule"]["minimum_learning_rate_factor"]),
    )
    bank_loader = TRAINER.deterministic_loader(
        bank_dataset, int(config["batch"]["validation_batch"]), workers
    )

    sampler_evidence = SimpleNamespace(
        train_filename=filenames,
        train_coordinates=coordinates,
        train_country=countries,
        train_descriptors={"fused": cached_teacher},
    )
    geographic = cv.geographic_positive_neighbours(sampler_evidence, config, device)
    row_weights = cv.anchor_row_weights(countries, config)

    resume_epoch = 0
    prior = sorted(args.output.glob("epoch_*.pt"))
    if args.resume and prior:
        checkpoint = torch.load(prior[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        resume_epoch = int(checkpoint["epoch"])
        print(f"RESUME at epoch {resume_epoch} from {prior[-1].name}", flush=True)
    elif prior and not args.resume:
        raise FileExistsError(f"{args.output} already holds checkpoints; pass --resume")

    if resume_epoch > 0:
        encoded = TRAINER.encode_dataset(ema.model, bank_loader, device, None)
        mining_descriptor = encoded["descriptor"].astype(np.float32)
    else:
        atomic_json(
            args.output / "run_config.json",
            {"config": config, "start_provenance": start_provenance, "images": 11_758},
        )
        encoded = TRAINER.encode_dataset(ema.model, bank_loader, device, None)
        if not np.array_equal(encoded["filename"].astype(str), filenames):
            raise RuntimeError("bank encode order differs from the row order")
        mining_descriptor = encoded["descriptor"].astype(np.float32)

    global_step = resume_epoch * steps_per_epoch
    started = time.perf_counter()
    for epoch in range(resume_epoch + 1, stop_after + 1):
        hard_rows, mining_report = cv.mine_regional_hard_negatives(
            mining_descriptor, coordinates, countries, config, device
        )
        limit = args.train_batches_limit
        sampler = cv.WeightedSpatialHardBatchSampler(
            geographic,
            hard_rows,
            physical_batch,
            seed + epoch,
            batches_per_epoch if limit is None else min(batches_per_epoch, limit),
            row_weights=row_weights,
        )
        loader = TRAINER.training_loader(train_dataset, sampler, workers, seed + epoch)
        epoch_stats, global_step = TRAINER.train_epoch(
            model,
            loader,
            spatial_objective,
            classification,
            cached_teacher,
            filename_to_position,
            optimizer,
            schedule,
            ema,
            config,
            device,
            global_step,
            limit,
        )
        elapsed = (time.perf_counter() - started) / 60.0
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "loss": epoch_stats["objective"],
                    "hard_negative_report": mining_report,
                    "elapsed_min": round(elapsed, 1),
                }
            ),
            flush=True,
        )
        atomic_torch_save(
            args.output / f"epoch_{epoch:03d}.pt",
            {
                "format": CHECKPOINT_FORMAT,
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "start_provenance": start_provenance,
            },
        )
        TRAINER.append_jsonl(
            args.output / "metrics.jsonl",
            {"epoch": epoch, "training": epoch_stats},
        )
        if epoch < stop_after:
            encoded = TRAINER.encode_dataset(ema.model, bank_loader, device, None)
            mining_descriptor = encoded["descriptor"].astype(np.float32)

    if stop_after == maximum_epochs:
        atomic_torch_save(
            args.output / "model_final.pt",
            {
                "format": CHECKPOINT_FORMAT,
                "epoch": maximum_epochs,
                "model": ema.model.state_dict(),
                "config": config,
                "start_provenance": start_provenance,
                "parameters": int(count_trainable_parameters(model)),
                "images": 11_758,
            },
        )
        print(f"wrote {args.output / 'model_final.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
