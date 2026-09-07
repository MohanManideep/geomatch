"""Shared retrieval utilities: teacher-cache loading, top-k search, spherical
distances. Used by the spatial-retrieval trainer and, through it, by the
country-aware finetune."""

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

ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_KM = 6371.0088
FUSED_DESCRIPTOR_WIDTH = 384
TEACHER_CACHE_FORMAT = "geomatch-teacher-cache-v1"
DEFAULT_TEACHER_CACHE = ROOT / "artifacts/teacher_cache"


@dataclass(frozen=True)
class FoldTeacher:
    """Frozen targets the retrieval objective distils toward, plus the fold's
    labels. Descriptors come from ``artifacts/teacher_cache``; every label,
    coordinate and filename is re-read from the committed fold assignments, so
    the cache cannot silently disagree with the split."""

    fold: int
    train_filename: np.ndarray
    train_coordinates: np.ndarray
    train_country: np.ndarray
    train_descriptors: dict
    val_filename: np.ndarray
    val_coordinates: np.ndarray
    val_country: np.ndarray
    val_descriptors: dict
    epoch0_hard_rows: list
    epoch0_hard_branch: np.ndarray
    cache_sha256: str
    descriptor_checkpoint_sha256: str
    mining_checkpoint_sha256: str


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


def load_fold_teacher(
    fold: int,
    directory: Path = DEFAULT_TEACHER_CACHE,
    sampling: dict | None = None,
) -> FoldTeacher:
    """Load one fold's frozen distillation targets.

    ``sampling``, when given, is the recipe's sampling block: the cached
    epoch-0 hard-negative rows were resolved under specific distance bands, so
    a recipe that changes them must not silently reuse the cached rows.
    Build the cache with ``src/build_teacher_cache.py``.
    """
    cache_path = directory / f"fold_{fold}.npz"
    manifest_path = directory / f"fold_{fold}.json"
    if not cache_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing fold-{fold} teacher cache in {directory}. "
            "Build it with: python src/build_teacher_cache.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != TEACHER_CACHE_FORMAT
        or int(manifest.get("fold", -1)) != fold
    ):
        raise ValueError(f"Fold-{fold} teacher manifest is invalid")
    cache_hash = sha256_file(cache_path)
    if manifest.get("cache_sha256") != cache_hash:
        raise ValueError(f"Fold-{fold} teacher cache hash differs from its manifest")
    if sampling is not None:
        recorded = manifest["sampling"]
        differing = [
            key for key, value in recorded.items() if sampling.get(key) != value
        ]
        if differing:
            raise ValueError(
                f"Fold-{fold} cached epoch-0 hard negatives were resolved under "
                f"different sampling settings ({differing}); rebuild the cache "
                "with src/build_teacher_cache.py"
            )

    with np.load(cache_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if str(arrays["format"].item()) != TEACHER_CACHE_FORMAT:
        raise ValueError(f"Fold-{fold} teacher cache format differs")
    if int(arrays["fold"].item()) != fold:
        raise ValueError(f"Fold-{fold} teacher cache belongs to another fold")

    training, validation = load_fold_assignments(fold)
    train_names = np.asarray(training["filename"], dtype=np.str_)
    val_names = np.asarray(validation["filename"], dtype=np.str_)
    if not np.array_equal(arrays["train_filename"].astype(str), train_names):
        raise RuntimeError(f"Fold-{fold} teacher cache training order differs")
    if not np.array_equal(arrays["val_filename"].astype(str), val_names):
        raise RuntimeError(f"Fold-{fold} teacher cache validation order differs")

    descriptors = {}
    for split, rows in (("train", len(training)), ("val", len(validation))):
        value = arrays[f"{split}_descriptor_fused"].astype(np.float32)
        if value.shape != (rows, FUSED_DESCRIPTOR_WIDTH):
            raise RuntimeError(f"Fold-{fold} {split} descriptor shape differs")
        if not np.isfinite(value).all():
            raise RuntimeError(f"Fold-{fold} {split} descriptor is non-finite")
        descriptors[split] = value

    candidates = arrays["epoch0_hard_candidates"].astype(np.int64)
    offsets = arrays["epoch0_hard_offsets"].astype(np.int64)
    branch = arrays["epoch0_hard_branch"].astype(np.int8)
    if len(offsets) != len(training) + 1 or len(branch) != len(training):
        raise RuntimeError(f"Fold-{fold} epoch-0 hard-negative rows are malformed")
    if offsets[0] != 0 or offsets[-1] != len(candidates):
        raise RuntimeError(f"Fold-{fold} epoch-0 hard-negative offsets are malformed")
    if candidates.size and (candidates.min() < 0 or candidates.max() >= len(training)):
        raise RuntimeError(f"Fold-{fold} epoch-0 hard negatives index outside the fold")
    hard_rows = [
        candidates[offsets[row] : offsets[row + 1]] for row in range(len(training))
    ]
    if min(len(row) for row in hard_rows) < 1:
        raise RuntimeError(f"Fold-{fold} has an empty epoch-0 hard-negative row")

    return FoldTeacher(
        fold=fold,
        train_filename=train_names,
        train_coordinates=training[["lat", "lng"]].to_numpy(dtype=np.float64),
        train_country=training["country_index"].to_numpy(dtype=np.int64),
        train_descriptors={"fused": descriptors["train"]},
        val_filename=val_names,
        val_coordinates=validation[["lat", "lng"]].to_numpy(dtype=np.float64),
        val_country=validation["country_index"].to_numpy(dtype=np.int64),
        val_descriptors={"fused": descriptors["val"]},
        epoch0_hard_rows=hard_rows,
        epoch0_hard_branch=branch,
        cache_sha256=cache_hash,
        descriptor_checkpoint_sha256=str(
            manifest["descriptor_source"]["checkpoint_sha256"]
        ),
        mining_checkpoint_sha256=str(manifest["mining_source"]["checkpoint_sha256"]),
    )
