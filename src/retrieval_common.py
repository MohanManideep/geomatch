"""Shared retrieval utilities: fold evidence loading, top-k search, spherical
distances. Used by the spatial-retrieval trainer (29) and, through it, by the
country-aware finetune (34)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_fold_assignments
from training import sha256_file

EARTH_RADIUS_KM = 6371.0088
DESCRIPTORS = ("global", "left", "right", "fused")
EXPECTED_DESCRIPTOR_WIDTHS = {"global": 256, "left": 256, "right": 256, "fused": 384}
EVIDENCE_FORMAT = "geomatch-regnet-cp-v2-failure-evidence-v1"


@dataclass(frozen=True)
class FoldEvidence:
    fold: int
    train_filename: np.ndarray
    train_coordinates: np.ndarray
    train_country: np.ndarray
    train_descriptors: dict
    val_filename: np.ndarray
    val_coordinates: np.ndarray
    val_country: np.ndarray
    val_descriptors: dict
    locked_prediction: np.ndarray
    evidence_sha256: str
    checkpoint_sha256: str


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def latlng_to_unit(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    latitude = np.radians(coordinates[..., 0])
    longitude = np.radians(coordinates[..., 1])
    cosine_latitude = np.cos(latitude)
    return np.stack(
        (
            cosine_latitude * np.cos(longitude),
            cosine_latitude * np.sin(longitude),
            np.sin(latitude),
        ),
        axis=-1,
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def matrix_topk(
    query: np.ndarray,
    bank: np.ndarray,
    top_k: int,
    device: torch.device,
    self_indices: np.ndarray | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError("Top-k matrices must have matching [rows, features] shapes")
    if not 1 <= top_k <= len(bank):
        raise ValueError("Top-k exceeds bank rows")
    if self_indices is not None and len(self_indices) != len(query):
        raise ValueError("Self-index vector length differs")
    bank_tensor = torch.from_numpy(normalize_rows(bank)).to(device)
    index_parts, score_parts = [], []
    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        value = torch.from_numpy(normalize_rows(query[start:stop])).to(device)
        score = value @ bank_tensor.T
        if self_indices is not None:
            local = torch.arange(stop - start, device=device)
            excluded = torch.from_numpy(self_indices[start:stop]).to(device)
            score[local, excluded] = -torch.inf
        top = score.topk(top_k, dim=1, largest=True, sorted=True)
        index_parts.append(top.indices.cpu().numpy())
        score_parts.append(top.values.cpu().numpy())
    del bank_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(index_parts), np.concatenate(score_parts).astype(np.float32)


def geographic_topk(
    coordinates: np.ndarray, top_k: int, device: torch.device
) -> np.ndarray:
    units = latlng_to_unit(coordinates).astype(np.float32)
    indices = np.arange(len(coordinates), dtype=np.int64)
    selected, _ = matrix_topk(
        units, units, top_k=top_k, device=device, self_indices=indices
    )
    return selected


def row_candidate_distances(
    truth: np.ndarray, candidate_coordinates: np.ndarray
) -> np.ndarray:
    truth_unit = latlng_to_unit(truth)
    candidate_unit = latlng_to_unit(candidate_coordinates)
    dots = np.sum(truth_unit[:, None, :] * candidate_unit, axis=2)
    return (EARTH_RADIUS_KM * np.arccos(np.clip(dots, -1.0, 1.0))).astype(np.float32)


def load_fold_evidence(directory: Path, fold: int) -> FoldEvidence:
    """Frozen teacher descriptors used to seed geographic-positive mining and
    the (now zero-weight) preservation term in the retrieval objective. Kept
    only so `descriptor_preservation_weight` stays wired for experimentation;
    the shipped recipe sets it to 0."""
    evidence_path = directory / f"fold_{fold}.npz"
    manifest_path = directory / f"fold_{fold}.json"
    if not evidence_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing fold-{fold} evidence in {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != EVIDENCE_FORMAT
        or int(manifest.get("fold", -1)) != fold
    ):
        raise ValueError(f"Fold-{fold} evidence manifest is invalid")
    evidence_hash = sha256_file(evidence_path)
    if manifest.get("evidence_sha256") != evidence_hash:
        raise ValueError(f"Fold-{fold} evidence hash differs")
    with np.load(evidence_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if str(arrays["format"].item()) != EVIDENCE_FORMAT:
        raise ValueError(f"Fold-{fold} NPZ format differs")
    if int(arrays["fold"].item()) != fold:
        raise ValueError(f"Fold-{fold} NPZ belongs to another fold")
    required = {
        "train_filename",
        "train_coordinates",
        "train_country",
        "val_filename",
        "val_coordinates",
        "val_country",
        "locked_prediction",
        *(f"train_descriptor_{name}" for name in DESCRIPTORS),
        *(f"val_descriptor_{name}" for name in DESCRIPTORS),
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise RuntimeError(f"Fold-{fold} evidence lacks {missing}")
    training, validation = load_fold_assignments(fold)
    if not np.array_equal(
        training["filename"].astype(str).to_numpy(),
        arrays["train_filename"].astype(str),
    ):
        raise RuntimeError(f"Fold-{fold} training evidence order differs")
    if not np.array_equal(
        validation["filename"].astype(str).to_numpy(),
        arrays["val_filename"].astype(str),
    ):
        raise RuntimeError(f"Fold-{fold} validation evidence order differs")
    train_descriptors, val_descriptors = {}, {}
    for name in DESCRIPTORS:
        train_value = arrays[f"train_descriptor_{name}"].astype(np.float32)
        val_value = arrays[f"val_descriptor_{name}"].astype(np.float32)
        width = EXPECTED_DESCRIPTOR_WIDTHS[name]
        if train_value.shape != (len(training), width):
            raise RuntimeError(f"Fold-{fold} training {name} descriptor shape differs")
        if val_value.shape != (len(validation), width):
            raise RuntimeError(
                f"Fold-{fold} validation {name} descriptor shape differs"
            )
        if not np.isfinite(train_value).all() or not np.isfinite(val_value).all():
            raise RuntimeError(f"Fold-{fold} {name} descriptor is non-finite")
        train_descriptors[name] = train_value
        val_descriptors[name] = val_value
    return FoldEvidence(
        fold=fold,
        train_filename=arrays["train_filename"].astype(str),
        train_coordinates=arrays["train_coordinates"].astype(np.float64),
        train_country=arrays["train_country"].astype(np.int64),
        train_descriptors=train_descriptors,
        val_filename=arrays["val_filename"].astype(str),
        val_coordinates=arrays["val_coordinates"].astype(np.float64),
        val_country=arrays["val_country"].astype(np.int64),
        val_descriptors=val_descriptors,
        locked_prediction=arrays["locked_prediction"].astype(np.float64),
        evidence_sha256=evidence_hash,
        checkpoint_sha256=str(manifest["checkpoint_sha256"]),
    )
