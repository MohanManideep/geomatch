"""End-to-end spatial retrieval fine-tuning on one official outer fold."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

import retrieval_common as COMMON
from data import (
    GeoMatchFoldDataset,
    ThreeViewTransform,
    build_fold_datasets,
    load_fold_assignments,
    read_json,
    seed_worker,
)
from joint_objective import DEFAULT_WEIGHTS, JointGeoClassificationObjective
from model import count_trainable_parameters
from retrieval_model import (
    EXPECTED_RETRIEVAL_PARAMETER_COUNT,
    GeoCPRegNetRetrieval,
    expected_retrieval_parameter_count,
    load_encoder_state_dict,
    retrieval_identity_error,
)
from retrieval_objective import (
    DescriptorMemoryQueue,
    build_spatial_objective,
    descriptor_preservation_loss,
    pairwise_haversine_km,
)
from scoring import distance_summary, move_batch
from training import (
    ModelEMA,
    WarmupCosineSchedule,
    atomic_json,
    atomic_torch_save,
    build_fingerprint,
    build_optimizer,
    capture_rng_state,
    restore_rng_state,
    set_seed,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/retrieval_finetune.json"
DEFAULT_IMAGES = Path("/var/tmp/luli38se-geomatch/data/geo_dataset/train")
DEFAULT_EVIDENCE = Path(
    "/var/tmp/luli38se-geomatch/outputs/regnet_cp_v2_failure_audit/evidence"
)
OUTPUT_ROOT = Path("/var/tmp/luli38se-geomatch/outputs")
COMMON_PATH = ROOT / "src/retrieval_common.py"
SOURCE_FORMAT = "geomatch-regnet-joint-finetune-checkpoint"
CHECKPOINT_FORMAT = "geomatch-regnet-spatial-retrieval-checkpoint"
CONFIG_FORMAT = "geomatch-regnet-spatial-retrieval-finetune"


class SpatialHardBatchSampler(Sampler[list[int]]):
    """Create anchor, two local-positive, one visual-hard-negative groups."""

    def __init__(
        self,
        geographic_neighbors: np.ndarray,
        hard_neighbors: list[np.ndarray],
        batch_size: int,
        seed: int,
        batches: int,
    ) -> None:
        if geographic_neighbors.ndim != 2 or len(geographic_neighbors) < batch_size:
            raise ValueError("Geographic neighbor matrix is invalid")
        if len(hard_neighbors) != len(geographic_neighbors):
            raise ValueError("Hard-neighbor rows differ")
        if batch_size % 4 != 0:
            raise ValueError("Physical batch must be divisible by the four-row group")
        if batches < 1:
            raise ValueError("Sampler must emit at least one batch")
        if any(len(value) < 1 for value in hard_neighbors):
            raise ValueError("Every anchor requires a hard-negative candidate")
        self.geographic_neighbors = geographic_neighbors.astype(np.int64)
        self.hard_neighbors = [value.astype(np.int64) for value in hard_neighbors]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.batches = int(batches)

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        rows = len(self.geographic_neighbors)
        groups = self.batch_size // 4
        anchor_order = rng.permutation(rows)
        anchor_position = 0
        for _ in range(self.batches):
            batch: list[int] = []
            used: set[int] = set()
            for _ in range(groups):
                selected_group = None
                attempts = 0
                while selected_group is None:
                    if anchor_position >= len(anchor_order):
                        anchor_order = rng.permutation(rows)
                        anchor_position = 0
                    anchor = int(anchor_order[anchor_position])
                    anchor_position += 1
                    attempts += 1
                    if attempts > rows * 2:
                        raise RuntimeError("Could not construct a unique spatial group")
                    if anchor in used:
                        continue
                    hard_order = rng.permutation(len(self.hard_neighbors[anchor]))
                    for hard_position in hard_order:
                        negative = int(self.hard_neighbors[anchor][int(hard_position)])
                        if negative in used or negative == anchor:
                            continue
                        blocked = used | {anchor, negative}
                        positive_values = [
                            int(value)
                            for value in self.geographic_neighbors[anchor]
                            if int(value) not in blocked
                        ]
                        if len(positive_values) < 2:
                            continue
                        positive_order = rng.permutation(len(positive_values))[:2]
                        positives = [
                            positive_values[int(position)]
                            for position in positive_order
                        ]
                        selected_group = (
                            anchor,
                            positives[0],
                            positives[1],
                            negative,
                        )
                        break
                used.update(selected_group)
                batch.extend(selected_group)
            if len(batch) != self.batch_size or len(set(batch)) != len(batch):
                raise RuntimeError("Spatial batch is not full and unique")
            yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the encoder end-to-end with continuous spatial retrieval"
    )
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--train-batches-limit", type=int)
    parser.add_argument("--validation-batches-limit", type=int)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_config(config: dict) -> dict:
    config = copy.deepcopy(config)
    if config.get("format") != CONFIG_FORMAT:
        raise ValueError("Unexpected retrieval configuration format")
    batch = config["batch"]
    physical = int(batch["physical_batch"])
    accumulation = int(batch["gradient_accumulation_steps"])
    if physical * accumulation != int(batch["effective_batch"]):
        raise ValueError("Physical and effective batch sizes disagree")
    if physical % 4 != 0:
        raise ValueError("Physical batch must be divisible by four")
    if int(config["checkpoint"]["fixed_final_epoch"]) < 1:
        raise ValueError("Fixed final epoch must be positive")
    if (
        config["checkpoint"]["selection_policy"]
        != "fixed_epoch_without_validation_selection"
    ):
        raise ValueError("This experiment forbids validation-selected stopping")
    objective = config["objective"]
    weights = [
        float(objective["spatial_retrieval_weight"]),
        float(objective["joint_classification_weight"]),
        float(objective["descriptor_preservation_weight"]),
    ]
    if min(weights) < 0.0 or abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(
            "Top-level objective weights must be nonnegative and sum to one"
        )
    sampling = config["sampling"]
    if sampling["group_layout"] != [
        "anchor",
        "geographic_positive_1",
        "geographic_positive_2",
        "raw_descriptor_hard_negative",
    ]:
        raise ValueError("Spatial group layout differs")
    if int(sampling["queries_per_group"]) != 3:
        raise ValueError("Exactly three query rows per spatial group are required")
    if int(sampling["geographic_positive_k"]) < 2:
        raise ValueError("At least two geographic positives are required")
    if int(sampling["geographic_positive_pool_k"]) < int(
        sampling["geographic_positive_k"]
    ):
        raise ValueError("Geographic positive pool is smaller than selected positives")
    if float(sampling["geographic_positive_maximum_km"]) <= 0.0:
        raise ValueError("Geographic positive radius must be positive")
    if int(sampling["hard_descriptor_k"]) < 1:
        raise ValueError("At least one hard descriptor neighbor is required")
    if not isinstance(sampling.get("refresh_hard_negatives_each_epoch"), bool):
        raise ValueError("Hard-negative refresh policy must be boolean")
    memory_capacity = int(objective["spatial_retrieval"]["memory_queue_capacity"])
    if memory_capacity < physical:
        raise ValueError("Memory queue must hold at least one physical batch")
    spatial = objective["spatial_retrieval"]
    local_weights = [
        float(spatial["global_weight"]),
        float(spatial["local_verification_weight"]),
    ]
    if min(local_weights) < 0.0 or abs(sum(local_weights) - 1.0) > 1e-9:
        raise ValueError("Global and local retrieval weights must sum to one")
    local_loss_weights = [
        float(spatial["local_hinge_weight"]),
        float(spatial["local_listwise_weight"]),
    ]
    if min(local_loss_weights) < 0.0 or abs(sum(local_loss_weights) - 1.0) > 1e-9:
        raise ValueError("Local hinge and listwise weights must sum to one")
    if float(spatial["local_embedding_temperature"]) <= 0.0:
        raise ValueError("Local embedding temperature must be positive")
    rerank_weight = float(config["evaluation"]["local_rerank_weight"])
    if not 0.0 <= rerank_weight <= 1.0:
        raise ValueError("Local rerank weight must lie in [0, 1]")
    local_features = int(config.get("model", {}).get("local_feature_dimension", 64))
    if expected_retrieval_parameter_count(local_features) > 5_000_000:
        raise ValueError("Configured local head exceeds the parameter limit")
    return config


def build_retrieval_fingerprint(
    fold: int,
    config_path: Path,
    source_path: Path,
    evidence_path: Path,
) -> dict:
    base = build_fingerprint(fold)
    files = {
        "config": config_path,
        "objective": ROOT / "src/retrieval_objective.py",
        "retrieval_model": ROOT / "src/retrieval_model.py",
        "trainer": ROOT / "src/spatial_retrieval_finetune.py",
        "common": COMMON_PATH,
        "source_checkpoint": source_path,
        "evidence": evidence_path,
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Fingerprint inputs missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in files.items()}
    combined = hashlib.sha256(
        json.dumps(
            {"base": base["combined_sha256"], "files": hashes}, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    return {
        "format": "geomatch-regnet-spatial-retrieval-fingerprint",
        "fold": fold,
        "base_project": base,
        "files": hashes,
        "combined_sha256": combined,
    }


def validate_source(
    checkpoint: dict,
    source_path: Path,
    evidence,
    config: dict,
    fold: int,
) -> None:
    if checkpoint.get("format") != SOURCE_FORMAT:
        raise ValueError("Unexpected joint source format")
    if int(checkpoint.get("epoch", -1)) != int(config["checkpoint"]["source_epoch"]):
        raise ValueError("Source epoch differs")
    if int(checkpoint.get("fingerprint", {}).get("fold", -1)) != fold:
        raise ValueError("Source belongs to another fold")
    if sha256_file(source_path) != evidence.checkpoint_sha256:
        raise ValueError("Source checkpoint differs from cached failure evidence")


def sampling_neighbors(
    evidence,
    config: dict,
    device: torch.device,
    descriptor_override: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    sampling = config["sampling"]
    geographic_k = int(sampling["geographic_positive_k"])
    geographic_pool_k = int(sampling["geographic_positive_pool_k"])
    hard_k = int(sampling["hard_descriptor_k"])
    geographic_pool = COMMON.geographic_topk(
        evidence.train_coordinates, geographic_pool_k, device
    )
    geographic_pool_distance = COMMON.row_candidate_distances(
        evidence.train_coordinates,
        evidence.train_coordinates[geographic_pool],
    )
    raw_teacher = COMMON.normalize_rows(evidence.train_descriptors["fused"])
    geographic_pool_similarity = np.sum(
        raw_teacher[:, None, :] * raw_teacher[geographic_pool], axis=2
    )
    maximum_positive_km = float(sampling["geographic_positive_maximum_km"])
    selected_geographic = []
    fallback_positive_rows = 0
    for row in range(len(geographic_pool)):
        local_positions = np.where(
            geographic_pool_distance[row] <= maximum_positive_km
        )[0]
        if len(local_positions) < geographic_k:
            fallback_positive_rows += 1
            local_positions = np.arange(len(geographic_pool[row]))
        similarity_order = local_positions[
            np.argsort(-geographic_pool_similarity[row, local_positions])
        ]
        selected_geographic.append(
            geographic_pool[row, similarity_order[:geographic_k]]
        )
    geographic = np.asarray(selected_geographic, dtype=np.int64)
    self_indices = np.arange(len(evidence.train_filename), dtype=np.int64)
    mining_descriptors = (
        evidence.train_descriptors["fused"]
        if descriptor_override is None
        else np.asarray(descriptor_override, dtype=np.float32)
    )
    if mining_descriptors.shape[0] != len(evidence.train_filename):
        raise ValueError("Mining descriptor rows differ from training evidence")
    hard, _ = COMMON.matrix_topk(
        mining_descriptors,
        mining_descriptors,
        hard_k,
        device,
        self_indices=self_indices,
    )
    geographic_distance = COMMON.row_candidate_distances(
        evidence.train_coordinates, evidence.train_coordinates[geographic]
    )
    hard_distance = COMMON.row_candidate_distances(
        evidence.train_coordinates, evidence.train_coordinates[hard]
    )
    threshold = float(sampling["hard_negative_minimum_km"])
    hard_rows = []
    fallback_rows = 0
    for row in range(len(hard)):
        valid = hard[row][hard_distance[row] >= threshold]
        if len(valid) == 0:
            fallback_rows += 1
            valid = hard[row][np.argsort(-hard_distance[row])[:1]]
        hard_rows.append(valid.astype(np.int64))
    report = {
        "rows": len(hard),
        "geographic_positive_k": geographic_k,
        "geographic_positive_pool_k": geographic_pool_k,
        "geographic_positive_maximum_km": maximum_positive_km,
        "rows_using_positive_fallback": fallback_positive_rows,
        "geographic_nearest_median_km": float(
            np.median(geographic_distance.min(axis=1))
        ),
        "geographic_nearest_within_50_pct": float(
            np.mean(geographic_distance.min(axis=1) <= 50.0) * 100.0
        ),
        "hard_descriptor_k": hard_k,
        "hard_negative_minimum_km": threshold,
        "rows_without_threshold_hard_negative": fallback_rows,
        "minimum_hard_candidates_per_row": min(len(value) for value in hard_rows),
        "mean_hard_candidates_per_row": float(
            np.mean([len(value) for value in hard_rows])
        ),
        "descriptor_source": (
            "raw_teacher" if descriptor_override is None else "current_ema"
        ),
    }
    return geographic, hard_rows, report


def deterministic_datasets(
    fold: int,
    images: Path,
) -> tuple[GeoMatchFoldDataset, GeoMatchFoldDataset]:
    normalization = read_json(ROOT / f"artifacts/normalization/fold_{fold}.json")
    data_config = read_json(ROOT / "configs/data.json")
    training_rows, validation_rows = load_fold_assignments(fold)
    transform = ThreeViewTransform(
        normalization["mean"], normalization["std"], training=False, config=data_config
    )
    return (
        GeoMatchFoldDataset(training_rows, images, transform),
        GeoMatchFoldDataset(validation_rows, images, transform),
    )


def deterministic_loader(dataset, batch_size: int, workers: int) -> DataLoader:
    generator = torch.Generator().manual_seed(503227)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def training_loader(
    dataset,
    sampler: SpatialHardBatchSampler,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


@torch.inference_mode()
def encode_dataset(
    model,
    loader,
    device: torch.device,
    maximum_batches: int | None,
    horizontal_flip_tta: bool = False,
) -> dict:
    model.eval()
    descriptors = []
    local_tokens = []
    country_probabilities = []
    coordinates = []
    countries = []
    filenames = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if maximum_batches is not None and batch_index >= maximum_batches:
            break
        views, targets, coordinate = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(*views)
            if horizontal_flip_tta:
                flipped_output = model(
                    torch.flip(views[0], dims=(-1,)),
                    torch.flip(views[2], dims=(-1,)),
                    torch.flip(views[1], dims=(-1,)),
                )
        retrieval_descriptor = output["retrieval_descriptor"].float()
        token_value = output["local_tokens"].float()
        country_probability = torch.softmax(output["country"].float(), dim=1)
        if horizontal_flip_tta:
            retrieval_descriptor = (
                retrieval_descriptor + flipped_output["retrieval_descriptor"].float()
            )
            flipped_tokens = (
                flipped_output["local_tokens"]
                .float()
                .reshape(len(token_value), 3, 4, 4, token_value.shape[-1])
            )
            flipped_tokens = torch.flip(flipped_tokens, dims=(3,)).reshape_as(
                token_value
            )
            token_value = torch.nn.functional.normalize(
                token_value + flipped_tokens, dim=2
            )
            country_probability = 0.5 * (
                country_probability
                + torch.softmax(flipped_output["country"].float(), dim=1)
            )
        descriptors.append(
            torch.nn.functional.normalize(retrieval_descriptor, dim=1).cpu().numpy()
        )
        local_tokens.append(token_value.cpu().numpy())
        country_probabilities.append(country_probability.cpu().numpy())
        coordinates.append(coordinate.cpu().numpy())
        countries.append(targets["country"].cpu().numpy())
        filenames.extend(str(value) for value in batch["filename"])
    if not descriptors:
        raise RuntimeError("Descriptor inference produced no rows")
    return {
        "descriptor": np.concatenate(descriptors).astype(np.float32),
        "local_tokens": np.concatenate(local_tokens).astype(np.float16),
        "country_probability": np.concatenate(country_probabilities).astype(np.float32),
        "coordinates": np.concatenate(coordinates).astype(np.float64),
        "country": np.concatenate(countries).astype(np.int64),
        "filename": np.asarray(filenames),
        "seconds": float(time.perf_counter() - started),
    }


def retrieval_metric_summary(distance: np.ndarray) -> dict:
    result = distance_summary(distance)
    result["within_50_pct"] = float(np.mean(distance <= 50.0) * 100.0)
    result["rows"] = int(len(distance))
    return result


def local_token_match_score(
    query: torch.Tensor, candidate: torch.Tensor
) -> torch.Tensor:
    """Symmetric late-interaction score for aligned or top-k image pairs."""

    if candidate.ndim == 3:
        similarity = torch.einsum("btd,bud->btu", query.float(), candidate.float())
    elif candidate.ndim == 4:
        similarity = torch.einsum("btd,bkud->bktu", query.float(), candidate.float())
    else:
        raise ValueError("Local candidate tokens must have rank three or four")
    query_coverage = similarity.max(dim=-1).values.mean(dim=-1)
    candidate_coverage = similarity.max(dim=-2).values.mean(dim=-1)
    return 0.5 * (query_coverage + candidate_coverage)


def local_verification_loss(tokens: torch.Tensor, margin: float) -> torch.Tensor:
    """Rank both sampled nearby images above the visual hard negative."""

    if tokens.ndim != 3 or len(tokens) % 4 != 0:
        raise ValueError("Local tokens must preserve four-row sampler groups")
    groups = tokens.reshape(-1, 4, tokens.shape[1], tokens.shape[2])
    anchor = groups[:, 0]
    positive_scores = torch.stack(
        (
            local_token_match_score(anchor, groups[:, 1]),
            local_token_match_score(anchor, groups[:, 2]),
        ),
        dim=1,
    )
    negative_score = local_token_match_score(anchor, groups[:, 3]).unsqueeze(1)
    return torch.relu(float(margin) + negative_score - positive_scores).mean()


def local_spatial_listwise_loss(
    tokens: torch.Tensor,
    coordinates: torch.Tensor,
    query_mask: torch.Tensor,
    teacher: torch.Tensor,
    radius_km: float,
    embedding_temperature: float,
    teacher_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise every query row against all local-token candidates in its batch."""

    similarity = torch.einsum("btd,cud->bctu", tokens.float(), tokens.float())
    score = 0.5 * (
        similarity.max(dim=-1).values.mean(dim=-1)
        + similarity.max(dim=-2).values.mean(dim=-1)
    )
    distance = pairwise_haversine_km(coordinates)
    identity = torch.eye(len(tokens), dtype=torch.bool, device=tokens.device)
    valid_candidate = ~identity
    positive = (distance <= float(radius_km)) & valid_candidate
    valid_query = query_mask & positive.any(dim=1)
    logits = score / float(embedding_temperature)
    logits = logits.masked_fill(~valid_candidate, -torch.inf)
    teacher_similarity = torch.nn.functional.normalize(teacher.float(), dim=1)
    teacher_similarity = teacher_similarity @ teacher_similarity.transpose(0, 1)
    target_logits = teacher_similarity / float(teacher_temperature)
    target_logits = target_logits.masked_fill(~positive, -torch.inf)
    target = torch.softmax(target_logits[valid_query], dim=1)
    log_probability = torch.log_softmax(logits[valid_query], dim=1)
    weighted_log_probability = torch.where(
        positive[valid_query], target * log_probability, torch.zeros_like(target)
    )
    loss = -weighted_log_probability.sum(dim=1).mean()
    valid_percent = valid_query.float().mean() * 100.0
    return loss, valid_percent


def rerank_with_local_tokens(
    query_tokens: np.ndarray,
    bank_tokens: np.ndarray,
    indices: np.ndarray,
    global_scores: np.ndarray,
    local_weight: float,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_parts = []
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        query = torch.from_numpy(query_tokens[start:stop].astype(np.float32)).to(device)
        candidate = torch.from_numpy(
            bank_tokens[indices[start:stop]].astype(np.float32)
        ).to(device)
        local_parts.append(local_token_match_score(query, candidate).cpu().numpy())
    local_scores = np.concatenate(local_parts).astype(np.float32)
    combined_scores = (1.0 - local_weight) * global_scores + local_weight * local_scores
    order = np.argsort(-combined_scores, axis=1)
    reranked_indices = np.take_along_axis(indices, order, axis=1)
    reranked_scores = np.take_along_axis(combined_scores, order, axis=1)
    reranked_local = np.take_along_axis(local_scores, order, axis=1)
    return reranked_indices, reranked_scores, reranked_local


def evaluate_retrieval(
    model,
    bank_loader,
    validation_loader,
    evidence,
    device: torch.device,
    validation_batches_limit: int | None,
    local_rerank_weight: float,
    top_k: int = 50,
) -> tuple[dict, pd.DataFrame]:
    bank = encode_dataset(model, bank_loader, device, maximum_batches=None)
    validation = encode_dataset(
        model,
        validation_loader,
        device,
        maximum_batches=validation_batches_limit,
    )
    if not np.array_equal(bank["filename"].astype(str), evidence.train_filename):
        raise RuntimeError("Deterministic bank order differs from cached evidence")
    expected_validation_names = evidence.val_filename[: len(validation["filename"])]
    if not np.array_equal(
        validation["filename"].astype(str), expected_validation_names
    ):
        raise RuntimeError("Validation descriptor order differs")
    indices, scores = COMMON.matrix_topk(
        validation["descriptor"], bank["descriptor"], top_k, device
    )
    global_indices = indices.copy()
    global_scores = scores.copy()
    indices, scores, local_scores = rerank_with_local_tokens(
        validation["local_tokens"],
        bank["local_tokens"],
        indices,
        scores,
        local_rerank_weight,
        device,
    )
    candidate_coordinates = bank["coordinates"][indices]
    candidate_distance = COMMON.row_candidate_distances(
        validation["coordinates"], candidate_coordinates
    )
    top1_distance = candidate_distance[:, 0]
    oracle_distance = candidate_distance.min(axis=1)
    global_top1_distance = COMMON.row_candidate_distances(
        validation["coordinates"], bank["coordinates"][global_indices[:, :1]]
    )[:, 0]
    predicted_country = bank["country"][indices[:, 0]]
    train_indices, _ = COMMON.matrix_topk(
        bank["descriptor"],
        bank["descriptor"],
        1,
        device,
        self_indices=np.arange(len(bank["descriptor"]), dtype=np.int64),
    )
    train_distance = COMMON.row_candidate_distances(
        bank["coordinates"], bank["coordinates"][train_indices]
    )[:, 0]
    teacher = COMMON.normalize_rows(evidence.train_descriptors["fused"])
    teacher_alignment = np.sum(bank["descriptor"] * teacher, axis=1)
    metrics = {
        "retrieval_top1": {
            **retrieval_metric_summary(top1_distance),
            "decoded_country_accuracy_pct": float(
                np.mean(predicted_country == validation["country"]) * 100.0
            ),
        },
        "global_retrieval_top1": retrieval_metric_summary(global_top1_distance),
        f"retrieval_top{top_k}_oracle": retrieval_metric_summary(oracle_distance),
        "train_leave_one_out_top1": retrieval_metric_summary(train_distance),
        "teacher_alignment": {
            "mean_cosine": float(np.mean(teacher_alignment)),
            "median_cosine": float(np.median(teacher_alignment)),
            "minimum_cosine": float(np.min(teacher_alignment)),
        },
        "inference_seconds": {
            "training_bank": bank["seconds"],
            "validation_queries": validation["seconds"],
        },
    }
    prediction = pd.DataFrame(
        {
            "filename": validation["filename"],
            "true_country": validation["country"],
            "true_lat": validation["coordinates"][:, 0],
            "true_lng": validation["coordinates"][:, 1],
            "predicted_country": predicted_country,
            "predicted_lat": candidate_coordinates[:, 0, 0],
            "predicted_lng": candidate_coordinates[:, 0, 1],
            "distance_km": top1_distance,
            f"top{top_k}_oracle_km": oracle_distance,
            "top1_similarity": scores[:, 0],
            "score_margin": scores[:, 0] - scores[:, 1],
            "top1_local_similarity": local_scores[:, 0],
            "global_top1_similarity": global_scores[:, 0],
            "global_top1_distance_km": global_top1_distance,
        }
    )
    return metrics, prediction


def objective_weights(config: dict) -> dict[str, float]:
    value = config["objective"]
    return {
        "spatial_retrieval": float(value["spatial_retrieval_weight"]),
        "joint_classification": float(value["joint_classification_weight"]),
        "descriptor_preservation": float(value["descriptor_preservation_weight"]),
    }


def train_epoch(
    model,
    loader,
    spatial_objective,
    classification_objective,
    cached_teacher: np.ndarray,
    filename_to_position: dict[str, int],
    optimizer,
    schedule,
    ema: ModelEMA,
    config: dict,
    device: torch.device,
    global_step: int,
    maximum_batches: int | None,
) -> tuple[dict, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = int(config["batch"]["gradient_accumulation_steps"])
    batches = (
        len(loader) if maximum_batches is None else min(len(loader), maximum_batches)
    )
    weights = objective_weights(config)
    rows = 0
    query_rows = 0
    sums = {
        "objective": 0.0,
        "spatial_retrieval": 0.0,
        "joint_classification": 0.0,
        "descriptor_preservation": 0.0,
        "retrieval_listwise": 0.0,
        "retrieval_triplet": 0.0,
        "local_verification": 0.0,
        "local_listwise": 0.0,
        "local_valid_query_pct": 0.0,
        "queries_with_radius_positive_pct": 0.0,
        "valid_triplet_queries_pct": 0.0,
        "mean_positive_count": 0.0,
        "candidate_count": 0.0,
        "memory_rows": 0.0,
    }
    memory = DescriptorMemoryQueue(
        int(config["objective"]["spatial_retrieval"]["memory_queue_capacity"])
    )
    gradient_norms = []
    started = time.perf_counter()
    current_group_size = accumulation
    for batch_index, batch in enumerate(loader):
        if batch_index >= batches:
            break
        position = batch_index % accumulation
        if position == 0:
            current_group_size = min(accumulation, batches - batch_index)
        views, targets, coordinates = move_batch(batch, device)
        filenames = [str(value) for value in batch["filename"]]
        teacher_positions = np.asarray(
            [filename_to_position[value] for value in filenames], dtype=np.int64
        )
        teacher = torch.from_numpy(cached_teacher[teacher_positions]).to(device)
        sample_ids = torch.from_numpy(teacher_positions).to(
            device=device, dtype=torch.long
        )
        query_mask = torch.tensor(
            [index % 4 < 3 for index in range(len(filenames))],
            dtype=torch.bool,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(*views)
            (
                memory_descriptors,
                memory_coordinates,
                memory_sample_ids,
                memory_teacher_descriptors,
            ) = memory.values()
            spatial_loss, spatial_terms = spatial_objective(
                outputs["retrieval_descriptor"],
                coordinates,
                query_mask,
                teacher_descriptors=teacher,
                sample_ids=sample_ids,
                memory_descriptors=memory_descriptors,
                memory_coordinates=memory_coordinates,
                memory_sample_ids=memory_sample_ids,
                memory_teacher_descriptors=memory_teacher_descriptors,
            )
            local_loss = local_verification_loss(
                outputs["local_tokens"],
                float(config["objective"]["spatial_retrieval"]["local_margin"]),
            )
            spatial_config = config["objective"]["spatial_retrieval"]
            local_listwise_loss, local_valid_percent = local_spatial_listwise_loss(
                outputs["local_tokens"],
                coordinates,
                query_mask,
                teacher,
                float(spatial_config["radius_km"]),
                float(spatial_config["local_embedding_temperature"]),
                float(spatial_config["teacher_target_temperature"]),
            )
            combined_local_loss = (
                float(spatial_config["local_hinge_weight"]) * local_loss
                + float(spatial_config["local_listwise_weight"]) * local_listwise_loss
            )
            combined_spatial_loss = (
                float(spatial_config["global_weight"]) * spatial_loss
                + float(spatial_config["local_verification_weight"])
                * combined_local_loss
            )
            classification_loss, _ = classification_objective(outputs, targets)
            preservation_loss = descriptor_preservation_loss(
                outputs["descriptor"], teacher
            )
            loss = (
                weights["spatial_retrieval"] * combined_spatial_loss
                + weights["joint_classification"] * classification_loss
                + weights["descriptor_preservation"] * preservation_loss
            )
        (loss / current_group_size).backward()
        memory.enqueue(
            outputs["retrieval_descriptor"], coordinates, sample_ids, teacher
        )
        batch_rows = len(filenames)
        batch_queries = int(query_mask.sum())
        rows += batch_rows
        query_rows += batch_queries
        sums["objective"] += float(loss.detach()) * batch_rows
        sums["spatial_retrieval"] += float(spatial_loss.detach()) * batch_rows
        sums["local_verification"] += float(local_loss.detach()) * batch_rows
        sums["local_listwise"] += float(local_listwise_loss.detach()) * batch_rows
        sums["local_valid_query_pct"] += (
            float(local_valid_percent.detach()) * batch_rows
        )
        sums["joint_classification"] += float(classification_loss.detach()) * batch_rows
        sums["descriptor_preservation"] += (
            float(preservation_loss.detach()) * batch_rows
        )
        sums["retrieval_listwise"] += (
            float(spatial_terms["listwise"].detach()) * batch_rows
        )
        sums["retrieval_triplet"] += (
            float(spatial_terms["triplet"].detach()) * batch_rows
        )
        for name in (
            "queries_with_radius_positive_pct",
            "valid_triplet_queries_pct",
            "mean_positive_count",
            "candidate_count",
            "memory_rows",
        ):
            sums[name] += float(spatial_terms[name].detach()) * batch_queries

        if position + 1 == current_group_size:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite gradient norm at step {global_step}")
            gradient_norms.append(float(gradient_norm))
            schedule.apply(global_step)
            optimizer.step()
            ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        if (batch_index + 1) % 50 == 0 or batch_index + 1 == batches:
            print(
                f"train_batch={batch_index + 1}/{batches} "
                f"loss={sums['objective'] / rows:.4f} global_step={global_step}",
                flush=True,
            )
    if rows == 0 or query_rows == 0 or not gradient_norms:
        raise RuntimeError("Retrieval training produced no updates")
    elapsed = time.perf_counter() - started
    return {
        "rows": rows,
        "query_rows": query_rows,
        "batches": batches,
        "optimizer_steps": len(gradient_norms),
        "seconds": elapsed,
        "images_per_second": rows / elapsed,
        "objective": sums["objective"] / rows,
        "loss_terms": {
            name: sums[name] / rows
            for name in (
                "spatial_retrieval",
                "joint_classification",
                "descriptor_preservation",
                "retrieval_listwise",
                "retrieval_triplet",
                "local_verification",
                "local_listwise",
                "local_valid_query_pct",
            )
        },
        "batch_spatial_health": {
            name: sums[name] / query_rows
            for name in (
                "queries_with_radius_positive_pct",
                "valid_triplet_queries_pct",
                "mean_positive_count",
                "candidate_count",
                "memory_rows",
            )
        },
        "memory_queue_final_rows": len(memory),
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "gradient_norm_max": float(np.max(gradient_norms)),
        "learning_rates": {
            group["name"]: group["lr"] for group in optimizer.param_groups
        },
    }, global_step


def checkpoint_payload(
    model,
    ema: ModelEMA,
    optimizer,
    epoch: int,
    global_step: int,
    fingerprint: dict,
    config: dict,
) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "fingerprint": fingerprint,
        "config": config,
        "selection_policy": config["checkpoint"]["selection_policy"],
        "rng_state": capture_rng_state(),
    }


def load_resume(
    path: Path, model, ema: ModelEMA, optimizer, fingerprint: dict, config: dict
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unexpected retrieval checkpoint format")
    if (
        checkpoint.get("fingerprint") != fingerprint
        or checkpoint.get("config") != config
    ):
        raise ValueError("Retrieval checkpoint provenance differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def self_test() -> int:
    set_seed(503227)
    coordinates = torch.tensor(
        [
            [52.5200, 13.4050],
            [52.5210, 13.4060],
            [48.8566, 2.3522],
            [48.8570, 2.3530],
            [41.9028, 12.4964],
            [41.9033, 12.4970],
            [59.3293, 18.0686],
            [59.3300, 18.0690],
        ],
        dtype=torch.float64,
    )
    descriptor = torch.randn(8, 16, requires_grad=True)
    query_mask = torch.tensor([True, True, True, False, True, True, True, False])
    objective = build_spatial_objective(
        {
            "embedding_temperature": 0.07,
            "radius_km": 50.0,
            "radius_target_weight": 0.75,
            "teacher_target_temperature": 0.07,
            "soft_distance_temperature_km": 50.0,
            "soft_distance_clip_km": 1000.0,
            "triplet_margin": 0.1,
            "negative_minimum_km": 200.0,
            "listwise_weight": 0.75,
            "triplet_weight": 0.25,
        }
    )
    sample_ids = torch.arange(8)
    loss, terms = objective(descriptor, coordinates, query_mask, sample_ids=sample_ids)
    memory = DescriptorMemoryQueue(capacity=16)
    memory.enqueue(descriptor, coordinates, sample_ids, descriptor.detach())
    (
        memory_descriptors,
        memory_coordinates,
        memory_ids,
        memory_teacher_descriptors,
    ) = memory.values()
    memory_loss, memory_terms = objective(
        descriptor,
        coordinates,
        query_mask,
        teacher_descriptors=descriptor.detach(),
        sample_ids=sample_ids,
        memory_descriptors=memory_descriptors,
        memory_coordinates=memory_coordinates,
        memory_sample_ids=memory_ids,
        memory_teacher_descriptors=memory_teacher_descriptors,
    )
    local_tokens = torch.nn.functional.normalize(
        torch.randn(8, 48, 64, requires_grad=True), dim=2
    )
    local_loss = local_verification_loss(local_tokens, margin=0.1)
    preservation = descriptor_preservation_loss(descriptor, descriptor.detach().clone())
    (loss + preservation + local_loss).backward()
    geographic = np.asarray(
        [[value for value in range(8) if value != row] for row in range(8)],
        dtype=np.int64,
    )
    hard = [
        np.asarray([value for value in range(8) if value != row], dtype=np.int64)
        for row in range(8)
    ]
    sampler = SpatialHardBatchSampler(
        geographic, hard, batch_size=8, seed=17, batches=1
    )
    batch = next(iter(sampler))
    model = GeoCPRegNetRetrieval()
    identity_error = retrieval_identity_error(model, torch.randn(8, 384))
    if len(batch) != len(set(batch)) or len(batch) != 8:
        raise RuntimeError("Spatial sampler self-test failed")
    if (
        not torch.isfinite(loss)
        or not torch.isfinite(memory_loss)
        or descriptor.grad is None
        or identity_error > 1e-7
        or int(memory_terms["memory_rows"]) != 8
    ):
        raise RuntimeError("Spatial objective self-test failed")
    print(
        json.dumps(
            {
                "batch": batch,
                "batch_unique": True,
                "loss": float(loss.detach()),
                "model_parameters": count_trainable_parameters(model),
                "retrieval_identity_error": identity_error,
                "memory_candidates": float(memory_terms["candidate_count"]),
                "local_verification_loss": float(local_loss.detach()),
                "preservation_identity_loss": float(preservation.detach()),
                "queries_with_radius_positive_pct": float(
                    terms["queries_with_radius_positive_pct"].detach()
                ),
                "status": "verified",
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    for value, name in (
        (args.train_batches_limit, "--train-batches-limit"),
        (args.validation_batches_limit, "--validation-batches-limit"),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    fold = int(args.fold)
    source_path = (
        OUTPUT_ROOT / f"regnet_cp_v2_joint/fold_{fold}/epoch_009.pt"
        if args.source is None
        else args.source
    ).resolve()
    output = (
        OUTPUT_ROOT / f"regnet_cp_v3_retrieval/fold_{fold}"
        if args.output is None
        else args.output
    ).resolve()
    config_path = args.config.resolve()
    config = validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    maximum_epochs = int(config["checkpoint"]["fixed_final_epoch"])
    stop_after = (
        maximum_epochs if args.stop_after_epoch is None else int(args.stop_after_epoch)
    )
    if not 1 <= stop_after <= maximum_epochs:
        raise ValueError(f"--stop-after-epoch must lie in [1,{maximum_epochs}]")
    last_path = output / "last.pt"
    if args.resume is None and last_path.exists():
        raise FileExistsError(f"Existing checkpoint found; pass --resume {last_path}")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    evidence = COMMON.load_fold_evidence(args.evidence.resolve(), fold)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    validate_source(source, source_path, evidence, config, fold)
    evidence_path = args.evidence.resolve() / f"fold_{fold}.npz"
    fingerprint = build_retrieval_fingerprint(
        fold, config_path, source_path, evidence_path
    )
    seed = int(config["seed"]) + fold * 10_000
    set_seed(seed)
    torch.backends.cudnn.benchmark = True

    train_dataset, _ = build_fold_datasets(
        fold,
        args.images.resolve(),
        ROOT / f"artifacts/normalization/fold_{fold}.json",
    )
    bank_dataset, validation_dataset = deterministic_datasets(
        fold, args.images.resolve()
    )
    if not np.array_equal(
        train_dataset.rows["filename"].astype(str).to_numpy(),
        evidence.train_filename,
    ):
        raise RuntimeError("Training dataset order differs from evidence")
    geographic, hard, sampling_report = sampling_neighbors(evidence, config, device)
    print(json.dumps({"sampling": sampling_report}, indent=2), flush=True)

    local_features = int(config.get("model", {}).get("local_feature_dimension", 64))
    model = GeoCPRegNetRetrieval(local_features=local_features).to(device)
    load_encoder_state_dict(model, source["model"])
    if count_trainable_parameters(model) != expected_retrieval_parameter_count(
        local_features
    ):
        raise RuntimeError("Retrieval parameter count differs")
    ema = ModelEMA(model, decay=float(config["ema"]["decay"]))
    spatial_objective = build_spatial_objective(
        config["objective"]["spatial_retrieval"]
    ).to(device)
    classification = JointGeoClassificationObjective(
        label_smoothing=float(config["objective"]["classification"]["label_smoothing"]),
        weights={
            name: float(value)
            for name, value in config["objective"]["classification"]["terms"].items()
        },
    ).to(device)
    if any(
        abs(classification.weights[name] - DEFAULT_WEIGHTS[name]) > 1e-12
        for name in DEFAULT_WEIGHTS
    ):
        raise ValueError(
            "Classification regularizer differs from the locked allocation"
        )
    optimizer = build_optimizer(model, config)
    physical_batch = int(config["batch"]["physical_batch"])
    batches_per_epoch = len(train_dataset) // physical_batch
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / int(config["batch"]["gradient_accumulation_steps"])
    )
    schedule = WarmupCosineSchedule(
        optimizer,
        total_steps=optimizer_steps_per_epoch * maximum_epochs,
        warmup_steps=optimizer_steps_per_epoch
        * int(config["schedule"]["warmup_epochs"]),
        minimum_factor=float(config["schedule"]["minimum_learning_rate_factor"]),
    )
    bank_loader = deterministic_loader(
        bank_dataset,
        int(config["batch"]["validation_batch"]),
        int(config["batch"]["workers"]),
    )
    validation_loader = deterministic_loader(
        validation_dataset,
        int(config["batch"]["validation_batch"]),
        int(config["batch"]["workers"]),
    )
    filename_to_position = {
        value: index for index, value in enumerate(evidence.train_filename)
    }
    cached_teacher = evidence.train_descriptors["fused"].astype(np.float32)

    start_epoch = 1
    global_step = 0
    if args.resume is None:
        initial_metrics, initial_predictions = evaluate_retrieval(
            ema.model,
            bank_loader,
            validation_loader,
            evidence,
            device,
            args.validation_batches_limit,
            float(config["evaluation"]["local_rerank_weight"]),
        )
        record = {
            "epoch": 0,
            "status": "initial",
            "validation": initial_metrics,
        }
        append_jsonl(output / "metrics.jsonl", record)
        COMMON.atomic_csv(output / "predictions_epoch_000.csv", initial_predictions)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        atomic_torch_save(
            output / "initial.pt",
            checkpoint_payload(model, ema, optimizer, 0, 0, fingerprint, config),
        )
    else:
        checkpoint = load_resume(
            args.resume.resolve(), model, ema, optimizer, fingerprint, config
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        print(f"resumed_epoch={start_epoch} global_step={global_step}", flush=True)

    atomic_json(output / "fingerprint.json", fingerprint)
    atomic_json(output / "run_config.json", config)
    atomic_json(output / "sampling_report.json", sampling_report)
    if start_epoch > stop_after:
        print(f"Nothing to do: checkpoint is already at epoch {start_epoch - 1}")
        return 0

    for epoch in range(start_epoch, stop_after + 1):
        epoch_started = time.perf_counter()
        if bool(config["sampling"]["refresh_hard_negatives_each_epoch"]):
            mining_bank = encode_dataset(
                ema.model, bank_loader, device, maximum_batches=None
            )
            geographic, hard, epoch_sampling = sampling_neighbors(
                evidence,
                config,
                device,
                descriptor_override=mining_bank["descriptor"],
            )
            epoch_sampling["epoch"] = epoch
            atomic_json(
                output / f"sampling_report_epoch_{epoch:03d}.json",
                epoch_sampling,
            )
            print(json.dumps({"sampling": epoch_sampling}, indent=2), flush=True)
        sampler = SpatialHardBatchSampler(
            geographic,
            hard,
            batch_size=physical_batch,
            seed=seed + epoch,
            batches=batches_per_epoch,
        )
        loader = training_loader(
            train_dataset,
            sampler,
            int(config["batch"]["workers"]),
            seed + epoch,
        )
        training, global_step = train_epoch(
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
            args.train_batches_limit,
        )
        del loader
        validation, predictions = evaluate_retrieval(
            ema.model,
            bank_loader,
            validation_loader,
            evidence,
            device,
            args.validation_batches_limit,
            float(config["evaluation"]["local_rerank_weight"]),
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "training": training,
            "validation": validation,
            "selection_policy": config["checkpoint"]["selection_policy"],
            "epoch_seconds": float(time.perf_counter() - epoch_started),
        }
        append_jsonl(output / "metrics.jsonl", record)
        atomic_json(output / "latest_metrics.json", record)
        COMMON.atomic_csv(output / f"predictions_epoch_{epoch:03d}.csv", predictions)
        payload = checkpoint_payload(
            model, ema, optimizer, epoch, global_step, fingerprint, config
        )
        atomic_torch_save(last_path, payload)
        atomic_torch_save(output / f"epoch_{epoch:03d}.pt", payload)
        print(json.dumps(record, indent=2, sort_keys=True), flush=True)

    summary = {
        "status": "completed",
        "fold": fold,
        "last_requested_epoch": stop_after,
        "fixed_final_epoch": maximum_epochs,
        "selection_policy": config["checkpoint"]["selection_policy"],
        "parameters": count_trainable_parameters(model),
        "output": str(output),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
