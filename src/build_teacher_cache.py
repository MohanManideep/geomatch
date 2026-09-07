"""Build the per-fold teacher cache that training reads.

The retrieval objective distils toward a set of frozen "teacher" descriptors:
they define the soft listwise targets (``teacher_target_temperature``) for both
the global and the local listwise terms, and they order the geographic-positive
pool. That happens whatever ``descriptor_preservation_weight`` is set to -- the
shipped recipe sets it to 0.0, which switches off only the extra cosine
preservation term, not the distillation targets.

Those descriptors were produced by two earlier encoders, neither of which is
part of the submitted pipeline:

* the fused 384-D descriptors come from the abandoned joint geo-cell
  classification model (``epoch_009.pt`` of each fold);
* the epoch-0 hard-negative mining ranking comes from the locked retrieval
  baseline (``epoch_006.pt`` of each fold). From epoch 1 onward the trainer
  re-mines from its own EMA, so this is the only place that encoder is read.

Shipping the two source caches would mean 840 MB of per-view descriptors,
per-cell log-probabilities and local tokens that training never touches. This
script distils them to the ~22 MB per fold that it does touch, verifies the
result, and records where every number came from. ``artifacts/teacher_cache/``
holds the output and is committed, so training runs from a clean checkout.

Build (needs the source caches):

    python src/build_teacher_cache.py --folds 0 1 2 3 4

Verify a committed cache against its manifest (needs nothing else):

    python src/build_teacher_cache.py --verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import retrieval_common as common
from data import load_fold_assignments
from training import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "artifacts/teacher_cache"
DEFAULT_CONFIG = ROOT / "configs/final_recipe.json"
DEFAULT_DESCRIPTOR_SOURCE = Path(
    "/var/tmp/luli38se-geomatch/outputs/regnet_cp_v2_failure_audit/evidence"
)
DEFAULT_MINING_SOURCE = Path(
    "/var/tmp/luli38se-geomatch/outputs/regnet_cp_v6_retrieval_features"
)

SAMPLING_KEYS = (
    "hard_descriptor_k",
    "hard_negative_same_country",
    "hard_negative_minimum_km",
    "hard_negative_maximum_km",
    "hard_negative_cross_country_fallback_minimum_km",
)


def epoch_zero_hard_rows(
    descriptors: np.ndarray,
    coordinates: np.ndarray,
    countries: np.ndarray,
    sampling: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the epoch-0 hard-negative candidate rows to plain indices.

    Mirrors ``mine_regional_hard_negatives`` exactly. Storing the resolved rows
    rather than the descriptors they came from keeps the cache small and makes
    the first epoch reproducible without the source encoder.
    """
    neighbours, _ = common.matrix_topk(
        descriptors.astype(np.float32),
        descriptors.astype(np.float32),
        int(sampling["hard_descriptor_k"]),
        device,
        self_indices=np.arange(len(descriptors), dtype=np.int64),
    )
    distance = common.row_candidate_distances(coordinates, coordinates[neighbours])
    minimum_km = float(sampling["hard_negative_minimum_km"])
    maximum_km = float(sampling["hard_negative_maximum_km"])
    fallback_km = float(sampling["hard_negative_cross_country_fallback_minimum_km"])
    same_country_only = bool(sampling["hard_negative_same_country"])

    candidates: list[np.ndarray] = []
    branches = np.empty(len(neighbours), dtype=np.int8)
    for row in range(len(neighbours)):
        candidate = neighbours[row]
        km = distance[row]
        same = countries[candidate] == countries[row]
        keep = (
            same & (km >= minimum_km) & (km <= maximum_km)
            if same_country_only
            else (km >= minimum_km) & (km <= maximum_km)
        )
        if keep.any():
            candidates.append(candidate[keep].astype(np.int32))
            branches[row] = 0
            continue
        cross = km >= fallback_km
        if cross.any():
            candidates.append(candidate[cross].astype(np.int32))
            branches[row] = 1
            continue
        candidates.append(candidate[np.argsort(-km)[:1]].astype(np.int32))
        branches[row] = 2

    offsets = np.zeros(len(candidates) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(value) for value in candidates])
    return np.concatenate(candidates), offsets, branches


def build_fold(
    fold: int,
    cache_directory: Path,
    descriptor_source: Path,
    mining_source: Path,
    sampling: dict,
    device: torch.device,
) -> dict:
    training, validation = load_fold_assignments(fold)
    train_names = np.asarray(training["filename"], dtype=np.str_)
    val_names = np.asarray(validation["filename"], dtype=np.str_)
    coordinates = training[["lat", "lng"]].to_numpy(dtype=np.float64)
    countries = training["country_index"].to_numpy(dtype=np.int64)

    descriptor_path = descriptor_source / f"fold_{fold}.npz"
    descriptor_manifest = json.loads(
        (descriptor_source / f"fold_{fold}.json").read_text(encoding="utf-8")
    )
    with np.load(descriptor_path, allow_pickle=False) as source:
        if not np.array_equal(source["train_filename"].astype(str), train_names):
            raise RuntimeError(f"fold {fold}: descriptor source is in another order")
        if not np.array_equal(source["val_filename"].astype(str), val_names):
            raise RuntimeError(
                f"fold {fold}: descriptor source validation order differs"
            )
        train_fused = source["train_descriptor_fused"].astype(np.float32)
        val_fused = source["val_descriptor_fused"].astype(np.float32)

    mining_path = mining_source / f"fold_{fold}.npz"
    mining_manifest = json.loads(
        (mining_source / f"fold_{fold}.json").read_text(encoding="utf-8")
    )
    with np.load(mining_path, allow_pickle=False) as source:
        if not np.array_equal(source["train_filename"].astype(str), train_names):
            raise RuntimeError(f"fold {fold}: mining source is in another order")
        mining_descriptor = source["train_descriptor"].astype(np.float32)

    candidates, offsets, branches = epoch_zero_hard_rows(
        mining_descriptor, coordinates, countries, sampling, device
    )

    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f"fold_{fold}.npz"
    temporary = cache_path.with_name(f"{cache_path.stem}.partial.npz")
    np.savez(
        temporary,
        format=np.array(common.TEACHER_CACHE_FORMAT),
        fold=np.array(fold, dtype=np.int64),
        train_filename=train_names,
        train_descriptor_fused=train_fused,
        val_filename=val_names,
        val_descriptor_fused=val_fused,
        epoch0_hard_candidates=candidates,
        epoch0_hard_offsets=offsets,
        epoch0_hard_branch=branches,
    )
    temporary.replace(cache_path)

    manifest = {
        "format": common.TEACHER_CACHE_FORMAT,
        "fold": fold,
        "cache_sha256": sha256_file(cache_path),
        "train_rows": int(len(train_names)),
        "val_rows": int(len(val_names)),
        "descriptor_features": int(train_fused.shape[1]),
        "sampling": {key: sampling[key] for key in SAMPLING_KEYS},
        "descriptor_source": {
            "role": "frozen listwise distillation targets and positive-pool ordering",
            "architecture": "joint geo-cell classification (abandoned)",
            "checkpoint": descriptor_manifest["checkpoint"],
            "checkpoint_sha256": descriptor_manifest["checkpoint_sha256"],
            "cache": str(descriptor_path),
            "cache_sha256": descriptor_manifest["evidence_sha256"],
        },
        "mining_source": {
            "role": "epoch-0 hard-negative ranking; later epochs re-mine from the EMA",
            "architecture": "locked three-view retrieval baseline",
            "checkpoint": mining_manifest["checkpoint"],
            "checkpoint_sha256": mining_manifest["checkpoint_sha256"],
            "cache": str(mining_path),
            "cache_sha256": mining_manifest["cache_sha256"],
        },
    }
    atomic_json(cache_directory / f"fold_{fold}.json", manifest)
    return {
        "fold": fold,
        "cache": str(cache_path),
        "megabytes": round(cache_path.stat().st_size / 1e6, 1),
        "epoch0_candidates": int(len(candidates)),
        "epoch0_mean_per_row": round(float(len(candidates) / len(train_names)), 1),
    }


def verify(cache_directory: Path, folds: list[int]) -> int:
    for fold in folds:
        teacher = common.load_fold_teacher(fold, cache_directory)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "train_rows": int(len(teacher.train_filename)),
                    "val_rows": int(len(teacher.val_filename)),
                    "cache_sha256": teacher.cache_sha256[:16],
                    "epoch0_rows": len(teacher.epoch0_hard_rows),
                    "status": "verified",
                }
            ),
            flush=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--descriptor-source", type=Path, default=DEFAULT_DESCRIPTOR_SOURCE
    )
    parser.add_argument("--mining-source", type=Path, default=DEFAULT_MINING_SOURCE)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        return verify(args.cache, args.folds)

    sampling = json.loads(args.config.read_text(encoding="utf-8"))["sampling"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in args.folds:
        report = build_fold(
            fold,
            args.cache,
            args.descriptor_source.resolve(),
            args.mining_source.resolve(),
            sampling,
            device,
        )
        print(json.dumps(report), flush=True)
    return verify(args.cache, args.folds)


if __name__ == "__main__":
    raise SystemExit(main())
