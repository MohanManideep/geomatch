"""Continue a pretrained retrieval encoder with country-aware negatives.

An earlier baseline's retrieval ranking was repaired only through post-hoc
tweaks on frozen features -- linear/residual/pairwise rerankers, zero-parameter
blends, whitening, local-token first-stage retrieval -- and none of it
generalized out of sample. A deep-shortlist audit showed why: a ``<=50`` km
candidate sits in the top-200 for ~80% of rows, the true country is in the
top-50 for 94%, yet within the weakest countries (Germany 35%, France 43%
oracle@50) the retrieved same-country images are scattered, not near the
query. That is a representation problem, so this experiment changes the
encoder itself instead.

It starts from a pretrained encoder's EMA weights and continues training with
two changes:

1. the fourth row of every sampled group is a **same-country, visually similar,
   120--900 km** hard negative (mined from the current EMA each epoch), instead
   of a mostly cross-country ">=200 km" negative -- forcing within-country
   spatial discrimination;
2. anchors from the weakest countries are oversampled.

Preservation toward the raw descriptor is halved and the backbone learning rate
is raised so the representation can actually move.  Everything else -- the
objective, the parameter count (4,869,911), the strict per-fold contract, the
fixed-epoch no-selection policy -- is unchanged.

Fold 0 is the pilot.  It advances to folds 1--4 only if its fixed-epoch
validation median beats ``checkpoint.pilot_advance_fold0_median_km``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as _tv_transforms
from torchvision.transforms import InterpolationMode as _tv_interp
from torchvision.transforms import functional as _tv_functional

from data import ISO_ORDER, ThreeViewTransform
from joint_objective import DEFAULT_WEIGHTS, JointGeoClassificationObjective
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
TRAINER_PATH = ROOT / "src/spatial_retrieval_finetune.py"
DEFAULT_CONFIG = ROOT / "configs/final_recipe.json"
DEFAULT_IMAGES = Path("/var/tmp/luli38se-geomatch/data/geo_dataset/train")
DEFAULT_EVIDENCE = Path(
    "/var/tmp/luli38se-geomatch/outputs/regnet_cp_v2_failure_audit/evidence"
)
OUTPUT_ROOT = Path("/var/tmp/luli38se-geomatch/outputs")
LOCKED_BASELINE_ROOT = OUTPUT_ROOT / "regnet_cp_v6_locked_oof"
DEFAULT_OUTPUT = OUTPUT_ROOT / "regnet_cp_v8_country_aware"
CONFIG_FORMAT = "geomatch-regnet-country-aware-retrieval-finetune"
CHECKPOINT_FORMAT = "geomatch-regnet-country-aware-retrieval-checkpoint"


class GlobalAugThreeViewTransform(ThreeViewTransform):
    """Anti-memorisation variant: add a random-resized-crop on the *global* view
    only (training mode). The base recipe feeds the global descriptor an
    unaltered 512px resize every epoch, so the encoder memorises training
    frames -- the honest reranker gate shows a ~15 km gap between an
    encoder-seen fold (37 km raw blend) and the same fold unseen (52 km).
    Left/right panel crops are already spatially varied and are left untouched.
    """

    def __init__(self, base: ThreeViewTransform, aug_cfg: dict) -> None:
        self.__dict__.update(base.__dict__)
        self._g_prob = float(aug_cfg["global_random_resized_crop_probability"])
        scale = tuple(float(x) for x in aug_cfg["global_random_resized_crop_scale"])
        ratio = tuple(
            float(x)
            for x in aug_cfg.get("global_random_resized_crop_ratio", [0.85, 1.18])
        )
        self._rrc = _tv_transforms.RandomResizedCrop(
            self.global_size,
            scale=scale,
            ratio=ratio,
            interpolation=_tv_interp.BILINEAR,
            antialias=True,
        )

    def __call__(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.training:
            if torch.rand(()).item() < self.color_probability:
                image = self.color_jitter(image)
            if torch.rand(()).item() < self.blur_probability:
                image = _tv_functional.gaussian_blur(
                    image, kernel_size=[5, 5], sigma=[0.1, 1.0]
                )
            left_end_fraction = float(torch.empty(()).uniform_(*self.left_range).item())
            right_start_fraction = float(
                torch.empty(()).uniform_(*self.right_range).item()
            )
        else:
            left_end_fraction = self.validation_left
            right_start_fraction = self.validation_right

        width, height = image.size
        left_end = max(1, min(width, int(round(width * left_end_fraction))))
        right_start = max(0, min(width - 1, int(round(width * right_start_fraction))))
        left = image.crop((0, 0, left_end, height))
        right = image.crop((right_start, 0, width, height))

        if self.training and torch.rand(()).item() < self._g_prob:
            cropped = self._rrc(image)
            value = (
                _tv_functional.pil_to_tensor(cropped)
                .to(dtype=torch.float32)
                .div_(255.0)
            )
            global_view = _tv_functional.normalize(value, mean=self.mean, std=self.std)
        else:
            global_view = self._tensor(image, self.global_size)

        return (
            global_view,
            self._tensor(left, self.local_size),
            self._tensor(right, self.local_size),
        )


def load_trainer():
    spec = importlib.util.spec_from_file_location(
        "_geomatch_retrieval_trainer", TRAINER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = load_trainer()
COMMON = TRAINER.COMMON


class WeightedSpatialHardBatchSampler(TRAINER.SpatialHardBatchSampler):
    """SpatialHardBatchSampler with per-row anchor sampling weights."""

    def __init__(self, *args, row_weights: np.ndarray, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        weights = np.asarray(row_weights, dtype=np.float64)
        if weights.shape != (len(self.geographic_neighbors),) or np.any(weights <= 0):
            raise ValueError("row_weights must be positive with one entry per row")
        self.row_probability = weights / weights.sum()

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        rows = len(self.geographic_neighbors)
        groups = self.batch_size // 4
        pool = rng.choice(rows, size=rows * 3, replace=True, p=self.row_probability)
        position = 0
        for _ in range(self.batches):
            batch: list[int] = []
            used: set[int] = set()
            for _ in range(groups):
                selected = None
                attempts = 0
                while selected is None:
                    if position >= len(pool):
                        pool = rng.choice(
                            rows, size=rows * 3, replace=True, p=self.row_probability
                        )
                        position = 0
                    anchor = int(pool[position])
                    position += 1
                    attempts += 1
                    if attempts > rows * 3:
                        raise RuntimeError("Could not construct a unique spatial group")
                    if anchor in used:
                        continue
                    hard_order = rng.permutation(len(self.hard_neighbors[anchor]))
                    for hp in hard_order:
                        negative = int(self.hard_neighbors[anchor][int(hp)])
                        if negative in used or negative == anchor:
                            continue
                        blocked = used | {anchor, negative}
                        positives = [
                            int(v)
                            for v in self.geographic_neighbors[anchor]
                            if int(v) not in blocked
                        ]
                        if len(positives) < 2:
                            continue
                        order = rng.permutation(len(positives))[:2]
                        selected = (
                            anchor,
                            positives[int(order[0])],
                            positives[int(order[1])],
                            negative,
                        )
                        break
                used.update(selected)
                batch.extend(selected)
            if len(batch) != self.batch_size or len(set(batch)) != len(batch):
                raise RuntimeError("Spatial batch is not full and unique")
            yield batch


def load_fold_cache_descriptor(fold: int, filenames: np.ndarray) -> np.ndarray:
    path = OUTPUT_ROOT / "regnet_cp_v6_retrieval_features" / f"fold_{fold}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing baseline feature cache for fold {fold} at {path}"
        )
    with np.load(path, allow_pickle=False) as source:
        cache_names = source["train_filename"].astype(str)
        descriptor = source["train_descriptor"].astype(np.float32)
    if not np.array_equal(cache_names, filenames):
        raise RuntimeError(
            "Baseline feature cache training order differs from fold evidence"
        )
    return descriptor


def geographic_positive_neighbours(
    evidence, config: dict, device: torch.device
) -> np.ndarray:
    sampling = config["sampling"]
    k = int(sampling["geographic_positive_k"])
    pool_k = int(sampling["geographic_positive_pool_k"])
    pool = COMMON.geographic_topk(evidence.train_coordinates, pool_k, device)
    pool_distance = COMMON.row_candidate_distances(
        evidence.train_coordinates, evidence.train_coordinates[pool]
    )
    teacher = COMMON.normalize_rows(evidence.train_descriptors["fused"])
    pool_similarity = np.sum(teacher[:, None, :] * teacher[pool], axis=2)
    maximum_km = float(sampling["geographic_positive_maximum_km"])
    selected = []
    for row in range(len(pool)):
        near = np.where(pool_distance[row] <= maximum_km)[0]
        if len(near) < k:
            near = np.arange(len(pool[row]))
        order = near[np.argsort(-pool_similarity[row, near])]
        selected.append(pool[row, order[:k]])
    return np.asarray(selected, dtype=np.int64)


def mine_regional_hard_negatives(
    descriptors: np.ndarray,
    coordinates: np.ndarray,
    countries: np.ndarray,
    config: dict,
    device: torch.device,
) -> tuple[list[np.ndarray], dict]:
    sampling = config["sampling"]
    hard_k = int(sampling["hard_descriptor_k"])
    min_km = float(sampling["hard_negative_minimum_km"])
    max_km = float(sampling["hard_negative_maximum_km"])
    fallback_km = float(sampling["hard_negative_cross_country_fallback_minimum_km"])
    same_country_only = bool(sampling["hard_negative_same_country"])
    self_indices = np.arange(len(descriptors), dtype=np.int64)
    neighbours, _ = COMMON.matrix_topk(
        descriptors.astype(np.float32),
        descriptors.astype(np.float32),
        hard_k,
        device,
        self_indices=self_indices,
    )
    neighbour_distance = COMMON.row_candidate_distances(
        coordinates, coordinates[neighbours]
    )
    rows: list[np.ndarray] = []
    same_country_used = 0
    cross_fallback_used = 0
    farthest_fallback_used = 0
    for row in range(len(neighbours)):
        cand = neighbours[row]
        km = neighbour_distance[row]
        same = countries[cand] == countries[row]
        keep = (
            same & (km >= min_km) & (km <= max_km)
            if same_country_only
            else (km >= min_km) & (km <= max_km)
        )
        if keep.any():
            rows.append(cand[keep].astype(np.int64))
            same_country_used += 1
            continue
        cross = km >= fallback_km
        if cross.any():
            rows.append(cand[cross].astype(np.int64))
            cross_fallback_used += 1
            continue
        rows.append(cand[np.argsort(-km)[:1]].astype(np.int64))
        farthest_fallback_used += 1
    report = {
        "rows": len(rows),
        "hard_descriptor_k": hard_k,
        "same_country_only": same_country_only,
        "distance_band_km": [min_km, max_km],
        "rows_with_same_country_regional_negative": same_country_used,
        "rows_using_cross_country_fallback": cross_fallback_used,
        "rows_using_farthest_fallback": farthest_fallback_used,
        "mean_candidates_per_row": float(np.mean([len(v) for v in rows])),
        "min_candidates_per_row": int(min(len(v) for v in rows)),
    }
    return rows, report


def anchor_row_weights(countries: np.ndarray, config: dict) -> np.ndarray:
    weights = np.ones(len(countries), dtype=np.float64)
    table = config["sampling"].get("failing_country_anchor_weight", {})
    iso_to_index = {iso: i for i, iso in enumerate(ISO_ORDER)}
    for iso, weight in table.items():
        if iso not in iso_to_index:
            raise ValueError(f"Unknown ISO in failing_country_anchor_weight: {iso}")
        weights[countries == iso_to_index[iso]] = float(weight)
    return weights


def load_start_model(
    fold: int, config: dict, device: torch.device
) -> tuple[GeoCPRegNetRetrieval, dict]:
    source = str(config.get("start_from", "regnet_cp_v6_locked_oof"))
    grid = int(config.get("local_grid_size", 4))
    model = GeoCPRegNetRetrieval(local_features=64, local_grid_size=grid).to(device)
    if grid != 4 and source != "ssl_backbone":
        raise ValueError(
            "local_grid_size != 4 requires start_from=ssl_backbone (earlier heads are 4x4)"
        )
    if source == "regnet_cp_v6_locked_oof":
        path = LOCKED_BASELINE_ROOT / f"fold_{fold}" / "epoch_006.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "geomatch-regnet-spatial-retrieval-checkpoint":
            raise ValueError(f"Unexpected locked-baseline checkpoint format at {path}")
        if int(checkpoint.get("epoch", -1)) != 6:
            raise ValueError(
                f"Expected the locked epoch-6 baseline checkpoint at {path}"
            )
        model.load_state_dict(checkpoint["ema"]["model"], strict=True)
        provenance = {
            "start_from": source,
            "checkpoint": str(path),
            "sha256": sha256_file(path),
        }
    elif source == "regnet_cp_v2_joint":
        source_epoch = int(config.get("source_epoch", 9))
        path = (
            OUTPUT_ROOT / f"regnet_cp_v2_joint/fold_{fold}/epoch_{source_epoch:03d}.pt"
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "geomatch-regnet-joint-finetune-checkpoint":
            raise ValueError(f"Unexpected joint checkpoint format at {path}")
        if int(checkpoint.get("epoch", -1)) != source_epoch:
            raise ValueError(f"Joint checkpoint at {path} is not epoch {source_epoch}")
        TRAINER.load_encoder_state_dict(model, checkpoint["model"])
        provenance = {
            "start_from": source,
            "checkpoint": str(path),
            "sha256": sha256_file(path),
            "source_epoch": source_epoch,
        }
    elif source == "ssl_backbone":
        path = Path(config["ssl_backbone_path"].replace("{fold}", str(fold)))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "geomatch-regnet-cp-ssl-byol-backbone-v1":
            raise ValueError(f"Unexpected SSL backbone format at {path}")
        if int(checkpoint.get("fold", -1)) != fold:
            raise ValueError(f"SSL backbone at {path} belongs to another fold")
        backbone_state = checkpoint["backbone"]
        incompatible = model.load_state_dict(backbone_state, strict=False)
        loaded = set(backbone_state)
        model_keys = set(model.state_dict())
        if not loaded.issubset(model_keys):
            raise RuntimeError(
                f"SSL backbone has keys not in the model: {sorted(loaded - model_keys)[:5]}"
            )
        if set(incompatible.unexpected_keys):
            raise RuntimeError(
                f"Unexpected SSL keys: {incompatible.unexpected_keys[:5]}"
            )
        expected_missing = {
            k
            for k in model_keys
            if not (k.startswith("stem.") or k.startswith("stages."))
        }
        actual_missing = set(incompatible.missing_keys)
        if actual_missing != expected_missing:
            raise RuntimeError(
                f"SSL load mismatch: only non-backbone keys should be missing. "
                f"unexpectedly missing backbone keys: {sorted(actual_missing - expected_missing)[:5]}; "
                f"unexpectedly loaded head keys: {sorted(expected_missing - actual_missing)[:5]}"
            )
        provenance = {
            "start_from": source,
            "checkpoint": str(path),
            "sha256": sha256_file(path),
            "ssl_epochs": int(checkpoint.get("epoch", -1)),
            "ssl_final_loss": checkpoint.get("final_loss"),
            "backbone_keys_loaded": len(loaded),
        }
    else:
        raise ValueError(f"Unknown start_from: {source}")
    if count_trainable_parameters(model) != EXPECTED_RETRIEVAL_PARAMETER_COUNT:
        raise RuntimeError(
            "Start model parameter count differs from the locked retrieval count"
        )
    return model, provenance


def summarise(distance: np.ndarray) -> dict:
    distance = np.asarray(distance, dtype=np.float64)
    return {
        "rows": int(distance.size),
        "median_km": float(np.median(distance)),
        "mean_km": float(np.mean(distance)),
        "within_25_pct": float(np.mean(distance <= 25.0) * 100.0),
        "within_50_pct": float(np.mean(distance <= 50.0) * 100.0),
        "within_100_pct": float(np.mean(distance <= 100.0) * 100.0),
    }


def run_fold(fold: int, config: dict, args, device: torch.device) -> dict:
    output = args.output / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"]) + fold * 10_000
    set_seed(seed)
    torch.backends.cudnn.benchmark = True

    evidence = COMMON.load_fold_evidence(args.evidence.resolve(), fold)
    normalization_path = ROOT / f"artifacts/normalization/fold_{fold}.json"
    train_dataset, _ = TRAINER.build_fold_datasets(
        fold, args.images.resolve(), normalization_path
    )
    global_aug = config.get("global_augmentation")
    if global_aug:
        train_dataset.transform = GlobalAugThreeViewTransform(
            train_dataset.transform, global_aug
        )
        print(
            f"[fold {fold}] global-view augmentation enabled: {json.dumps(global_aug)}",
            flush=True,
        )
    bank_dataset, validation_dataset = TRAINER.deterministic_datasets(
        fold, args.images.resolve()
    )
    if not np.array_equal(
        train_dataset.rows["filename"].astype(str).to_numpy(), evidence.train_filename
    ):
        raise RuntimeError("Training dataset order differs from evidence")

    model, start_provenance = load_start_model(fold, config, device)
    ema = ModelEMA(model, decay=float(config["ema"]["decay"]))
    spatial_objective = build_spatial_objective(
        config["objective"]["spatial_retrieval"]
    ).to(device)
    classification = JointGeoClassificationObjective(
        label_smoothing=float(config["objective"]["classification"]["label_smoothing"]),
        weights={
            n: float(v)
            for n, v in config["objective"]["classification"]["terms"].items()
        },
    ).to(device)
    if any(
        abs(classification.weights[n] - DEFAULT_WEIGHTS[n]) > 1e-12
        for n in DEFAULT_WEIGHTS
    ):
        raise ValueError(
            "Classification regularizer differs from the locked allocation"
        )
    optimizer = build_optimizer(model, config)

    physical_batch = int(config["batch"]["physical_batch"])
    accumulation = int(config["batch"]["gradient_accumulation_steps"])
    batches_per_epoch = len(train_dataset) // physical_batch
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    maximum_epochs = int(config["checkpoint"]["fixed_final_epoch"])
    schedule = WarmupCosineSchedule(
        optimizer,
        total_steps=steps_per_epoch * maximum_epochs,
        warmup_steps=steps_per_epoch * int(config["schedule"]["warmup_epochs"]),
        minimum_factor=float(config["schedule"]["minimum_learning_rate_factor"]),
    )
    workers = int(config["batch"]["workers"])
    bank_loader = TRAINER.deterministic_loader(
        bank_dataset, int(config["batch"]["validation_batch"]), workers
    )
    validation_loader = TRAINER.deterministic_loader(
        validation_dataset, int(config["batch"]["validation_batch"]), workers
    )
    filename_to_position = {v: i for i, v in enumerate(evidence.train_filename)}
    cached_teacher = evidence.train_descriptors["fused"].astype(np.float32)
    geographic = geographic_positive_neighbours(evidence, config, device)
    row_weights = anchor_row_weights(evidence.train_country, config)
    rerank_weight = float(config["evaluation"]["local_rerank_weight"])
    top_k = int(config["evaluation"]["shortlist_top_k"])

    stop_after = (
        maximum_epochs if args.stop_after_epoch is None else int(args.stop_after_epoch)
    )
    train_limit = args.train_batches_limit
    val_limit = args.validation_batches_limit

    def evaluate(tag_model, epoch: int) -> tuple[dict, "object"]:
        metrics, predictions = TRAINER.evaluate_retrieval(
            tag_model,
            bank_loader,
            validation_loader,
            evidence,
            device,
            val_limit,
            rerank_weight,
            top_k=top_k,
        )
        COMMON.atomic_csv(output / f"predictions_epoch_{epoch:03d}.csv", predictions)
        return metrics, predictions

    resume_epoch = 0
    if args.resume:
        prior = sorted(output.glob("epoch_*.pt"))
        if prior:
            ck = torch.load(prior[-1], map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            ema.load_state_dict(ck["ema"])
            resume_epoch = int(ck["epoch"])
            print(
                f"[fold {fold}] RESUME at epoch {resume_epoch} from {prior[-1].name}",
                flush=True,
            )

    if resume_epoch > 0:
        lines = [
            l for l in (output / "metrics.jsonl").read_text().splitlines() if l.strip()
        ]
        history = [json.loads(l) for l in lines]
        history = [h for h in history if int(h.get("epoch", -1)) <= resume_epoch]
        initial_metrics = history[0]["validation"]
        enc = TRAINER.encode_dataset(
            ema.model, bank_loader, device, maximum_batches=None
        )
        mining_descriptor = enc["descriptor"].astype(np.float32)
    else:
        history = []
        initial_metrics, _ = evaluate(ema.model, 0)
        history.append({"epoch": 0, "status": "start", "validation": initial_metrics})
        TRAINER.append_jsonl(output / "metrics.jsonl", history[-1])
        print(
            json.dumps(
                {"fold": fold, "epoch": 0, "validation": initial_metrics}, indent=2
            ),
            flush=True,
        )
        mining_descriptor = load_fold_cache_descriptor(fold, evidence.train_filename)

    global_step = resume_epoch * steps_per_epoch
    for epoch in range(resume_epoch + 1, stop_after + 1):
        hard_rows, mining_report = mine_regional_hard_negatives(
            mining_descriptor,
            evidence.train_coordinates,
            evidence.train_country,
            config,
            device,
        )
        atomic_json(
            output / f"hard_negative_report_epoch_{epoch:03d}.json", mining_report
        )
        sampler = WeightedSpatialHardBatchSampler(
            geographic,
            hard_rows,
            physical_batch,
            seed + epoch,
            (
                batches_per_epoch
                if train_limit is None
                else min(batches_per_epoch, train_limit)
            ),
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
            train_limit,
        )
        metrics, _ = evaluate(ema.model, epoch)
        record = {
            "epoch": epoch,
            "training": epoch_stats,
            "validation": metrics,
            "hard_negative_report": mining_report,
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "val_reranked": metrics["retrieval_top1"],
                    "val_oracle": metrics.get(f"retrieval_top{top_k}_oracle", {}),
                },
                indent=2,
            ),
            flush=True,
        )
        atomic_torch_save(
            output / f"epoch_{epoch:03d}.pt",
            {
                "format": CHECKPOINT_FORMAT,
                "fold": fold,
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "config": config,
                "start_provenance": start_provenance,
            },
        )
        # jsonl AFTER the checkpoint so --resume never sees an epoch it can't reload
        TRAINER.append_jsonl(output / "metrics.jsonl", record)
        # refresh mining space from the current EMA for the next epoch
        if (
            bool(config["sampling"]["refresh_hard_negatives_each_epoch"])
            and epoch < stop_after
        ):
            encoded = TRAINER.encode_dataset(
                ema.model, bank_loader, device, maximum_batches=None
            )
            if not np.array_equal(
                encoded["filename"].astype(str), evidence.train_filename
            ):
                raise RuntimeError("EMA bank encode order differs from evidence")
            mining_descriptor = encoded["descriptor"].astype(np.float32)

    final_metrics = history[-1]["validation"]["retrieval_top1"]
    summary = {
        "fold": fold,
        "epochs": stop_after,
        "parameters": int(count_trainable_parameters(model)),
        "start_provenance": start_provenance,
        "start_val": initial_metrics["retrieval_top1"],
        "final_val": final_metrics,
        "final_oracle": history[-1]["validation"].get(
            f"retrieval_top{top_k}_oracle", {}
        ),
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--train-batches-limit", type=int)
    parser.add_argument("--validation-batches-limit", type=int)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="override config['seed'] (for training independent ensemble members)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume a fold from its latest epoch_NNN.pt in the output dir",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("format") != CONFIG_FORMAT:
        raise ValueError("Unexpected configuration format")
    if args.seed_base is not None:
        config["seed"] = int(args.seed_base)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)

    folds = args.folds if args.folds is not None else [0]
    pilot_gate = float(config["checkpoint"]["pilot_advance_fold0_median_km"])
    summaries = []
    for fold in folds:
        print(f"\n===== fold {fold} =====", flush=True)
        summary = run_fold(fold, config, args, device)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
        if fold == 0 and args.folds is None:
            median = summary["final_val"]["median_km"]
            if median > pilot_gate:
                print(
                    f"\nPILOT GATE NOT MET: fold-0 median {median:.2f} km > {pilot_gate} km. "
                    f"Not advancing to folds 1-4.",
                    flush=True,
                )
                atomic_json(
                    args.output / "pilot_result.json",
                    {"passed": False, "fold0": summary},
                )
                return 1
            print(
                f"\nPILOT GATE MET: fold-0 median {median:.2f} km <= {pilot_gate} km.",
                flush=True,
            )
            for extra in (1, 2, 3, 4):
                summaries.append(run_fold(extra, config, args, device))

    atomic_json(
        args.output / "crossfit_summary.json",
        {"folds": [s["fold"] for s in summaries], "per_fold": summaries},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
