"""Continuous spatial retrieval objective for end-to-end retrieval fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

EARTH_RADIUS_KM = 6371.0088


def coordinates_to_unit(coordinates: Tensor) -> Tensor:
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [batch, 2]")
    value = coordinates.float()
    latitude = torch.deg2rad(value[:, 0])
    longitude = torch.deg2rad(value[:, 1])
    cosine_latitude = torch.cos(latitude)
    return torch.stack(
        (
            cosine_latitude * torch.cos(longitude),
            cosine_latitude * torch.sin(longitude),
            torch.sin(latitude),
        ),
        dim=1,
    )


def pairwise_haversine_km(coordinates: Tensor) -> Tensor:
    return cross_haversine_km(coordinates, coordinates)


def cross_haversine_km(left: Tensor, right: Tensor) -> Tensor:
    left_units = coordinates_to_unit(left)
    right_units = coordinates_to_unit(right)
    dots = left_units @ right_units.T
    return EARTH_RADIUS_KM * torch.acos(dots.clamp(-1.0, 1.0))


class DescriptorMemoryQueue:
    """Bounded FIFO of detached embeddings used as cross-batch candidates."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("memory queue capacity cannot be negative")
        self.capacity = int(capacity)
        self.descriptors: Tensor | None = None
        self.coordinates: Tensor | None = None
        self.sample_ids: Tensor | None = None
        self.teacher_descriptors: Tensor | None = None

    def __len__(self) -> int:
        return 0 if self.sample_ids is None else int(self.sample_ids.numel())

    def values(
        self,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        return (
            self.descriptors,
            self.coordinates,
            self.sample_ids,
            self.teacher_descriptors,
        )

    @torch.no_grad()
    def enqueue(
        self,
        descriptors: Tensor,
        coordinates: Tensor,
        sample_ids: Tensor,
        teacher_descriptors: Tensor,
    ) -> None:
        if self.capacity == 0:
            return
        if descriptors.ndim != 2:
            raise ValueError("queued descriptors must have shape [batch, features]")
        if coordinates.shape != (len(descriptors), 2):
            raise ValueError("queued coordinates differ from descriptors")
        if sample_ids.shape != (len(descriptors),):
            raise ValueError("queued sample ids differ from descriptors")
        if teacher_descriptors.shape != descriptors.shape:
            raise ValueError("queued teacher descriptors differ")
        incoming = (
            F.normalize(descriptors.detach().float(), dim=1),
            coordinates.detach().float(),
            sample_ids.detach().long(),
            F.normalize(teacher_descriptors.detach().float(), dim=1),
        )
        if self.descriptors is None:
            descriptors_value, coordinates_value, ids_value, teacher_value = incoming
        else:
            descriptors_value = torch.cat((self.descriptors, incoming[0]), dim=0)
            coordinates_value = torch.cat((self.coordinates, incoming[1]), dim=0)
            ids_value = torch.cat((self.sample_ids, incoming[2]), dim=0)
            teacher_value = torch.cat((self.teacher_descriptors, incoming[3]), dim=0)
        self.descriptors = descriptors_value[-self.capacity :]
        self.coordinates = coordinates_value[-self.capacity :]
        self.sample_ids = ids_value[-self.capacity :]
        self.teacher_descriptors = teacher_value[-self.capacity :]


class SpatialRetrievalObjective(nn.Module):
    """Optimize continuous distance ordering for spatially structured batches."""

    def __init__(
        self,
        embedding_temperature: float,
        radius_km: float,
        radius_target_weight: float,
        teacher_target_temperature: float,
        soft_distance_temperature_km: float,
        soft_distance_clip_km: float,
        triplet_margin: float,
        negative_minimum_km: float,
        listwise_weight: float,
        triplet_weight: float,
    ) -> None:
        super().__init__()
        for value, name in (
            (embedding_temperature, "embedding_temperature"),
            (teacher_target_temperature, "teacher_target_temperature"),
            (radius_km, "radius_km"),
            (soft_distance_temperature_km, "soft_distance_temperature_km"),
            (soft_distance_clip_km, "soft_distance_clip_km"),
            (negative_minimum_km, "negative_minimum_km"),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= radius_target_weight <= 1.0:
            raise ValueError("radius_target_weight must lie in [0,1]")
        if triplet_margin < 0.0:
            raise ValueError("triplet_margin cannot be negative")
        if listwise_weight < 0.0 or triplet_weight < 0.0:
            raise ValueError("retrieval component weights cannot be negative")
        if abs(listwise_weight + triplet_weight - 1.0) > 1e-9:
            raise ValueError("retrieval component weights must sum to one")
        self.embedding_temperature = float(embedding_temperature)
        self.radius_km = float(radius_km)
        self.radius_target_weight = float(radius_target_weight)
        self.teacher_target_temperature = float(teacher_target_temperature)
        self.soft_distance_temperature_km = float(soft_distance_temperature_km)
        self.soft_distance_clip_km = float(soft_distance_clip_km)
        self.triplet_margin = float(triplet_margin)
        self.negative_minimum_km = float(negative_minimum_km)
        self.listwise_weight = float(listwise_weight)
        self.triplet_weight = float(triplet_weight)

    def forward(
        self,
        descriptors: Tensor,
        coordinates: Tensor,
        query_mask: Tensor,
        teacher_descriptors: Tensor | None = None,
        sample_ids: Tensor | None = None,
        memory_descriptors: Tensor | None = None,
        memory_coordinates: Tensor | None = None,
        memory_sample_ids: Tensor | None = None,
        memory_teacher_descriptors: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if descriptors.ndim != 2:
            raise ValueError("descriptors must have shape [batch, features]")
        if coordinates.shape != (descriptors.shape[0], 2):
            raise ValueError("coordinate and descriptor batches differ")
        if (
            query_mask.shape != (descriptors.shape[0],)
            or query_mask.dtype != torch.bool
        ):
            raise ValueError("query_mask must be boolean with shape [batch]")
        if not query_mask.any():
            raise ValueError("query_mask selects no rows")
        if sample_ids is None:
            sample_ids = torch.arange(
                len(descriptors), dtype=torch.long, device=descriptors.device
            )
        if sample_ids.shape != (len(descriptors),):
            raise ValueError("sample_ids must have shape [batch]")
        if (
            teacher_descriptors is not None
            and teacher_descriptors.shape != descriptors.shape
        ):
            raise ValueError("teacher descriptors differ from student descriptors")
        memory_values = (
            memory_descriptors,
            memory_coordinates,
            memory_sample_ids,
            memory_teacher_descriptors,
        )
        if any(value is None for value in memory_values) and not all(
            value is None for value in memory_values
        ):
            raise ValueError(
                "memory descriptors, coordinates and ids must be supplied together"
            )

        embedding = F.normalize(descriptors.float(), dim=1)
        candidate_embedding = embedding
        candidate_coordinates = coordinates
        candidate_ids = sample_ids
        teacher_embedding = (
            None
            if teacher_descriptors is None
            else F.normalize(teacher_descriptors.detach().float(), dim=1)
        )
        candidate_teacher = teacher_embedding
        memory_rows = 0
        if memory_descriptors is not None:
            if (
                memory_descriptors.ndim != 2
                or memory_descriptors.shape[1] != descriptors.shape[1]
            ):
                raise ValueError("memory descriptor shape differs")
            if memory_coordinates.shape != (len(memory_descriptors), 2):
                raise ValueError("memory coordinate shape differs")
            if memory_sample_ids.shape != (len(memory_descriptors),):
                raise ValueError("memory sample id shape differs")
            if memory_teacher_descriptors.shape != memory_descriptors.shape:
                raise ValueError("memory teacher descriptor shape differs")
            memory_rows = len(memory_descriptors)
            candidate_embedding = torch.cat(
                (embedding, F.normalize(memory_descriptors.float(), dim=1)), dim=0
            )
            candidate_coordinates = torch.cat(
                (coordinates, memory_coordinates.to(coordinates)), dim=0
            )
            candidate_ids = torch.cat(
                (sample_ids, memory_sample_ids.to(sample_ids)), dim=0
            )
            if teacher_embedding is None:
                raise ValueError("current teacher descriptors are required with memory")
            candidate_teacher = torch.cat(
                (
                    teacher_embedding,
                    F.normalize(memory_teacher_descriptors.float(), dim=1),
                ),
                dim=0,
            )
        queries = torch.where(query_mask)[0]
        query_similarity = embedding[queries] @ candidate_embedding.T
        query_distance = cross_haversine_km(coordinates[queries], candidate_coordinates)
        query_excluded = sample_ids[queries, None] == candidate_ids[None, :]

        scores = query_similarity / self.embedding_temperature
        scores = scores.masked_fill(query_excluded, -torch.inf)
        positive = (query_distance <= self.radius_km) & ~query_excluded
        positive_count = positive.sum(dim=1, keepdim=True)

        soft_logits = (
            -query_distance.clamp_max(self.soft_distance_clip_km)
            / self.soft_distance_temperature_km
        )
        soft_logits = soft_logits.masked_fill(query_excluded, -torch.inf)
        soft_target = torch.softmax(soft_logits, dim=1)
        radius_target = positive.float() / positive_count.clamp_min(1)
        if teacher_embedding is not None:
            teacher_similarity = teacher_embedding[queries] @ candidate_teacher.T
            teacher_logits = teacher_similarity / self.teacher_target_temperature
            teacher_logits = teacher_logits.masked_fill(~positive, -torch.inf)
            teacher_radius_target = torch.softmax(teacher_logits, dim=1)
            radius_target = torch.where(
                positive_count > 0, teacher_radius_target, radius_target
            )
        blended = (
            self.radius_target_weight * radius_target
            + (1.0 - self.radius_target_weight) * soft_target
        )
        target = torch.where(positive_count > 0, blended, soft_target)
        log_probability = torch.log_softmax(scores, dim=1).masked_fill(
            query_excluded, 0.0
        )
        listwise = -(target * log_probability).sum(dim=1).mean()

        has_positive = positive.any(dim=1)
        negative = (query_distance >= self.negative_minimum_km) & ~query_excluded
        has_negative = negative.any(dim=1)
        valid_triplet = has_positive & has_negative
        positive_similarity = (
            query_similarity.masked_fill(~positive, -torch.inf).max(dim=1).values
        )
        negative_similarity = (
            query_similarity.masked_fill(~negative, -torch.inf).max(dim=1).values
        )
        if valid_triplet.any():
            triplet = F.relu(
                self.triplet_margin
                + negative_similarity[valid_triplet]
                - positive_similarity[valid_triplet]
            ).mean()
        else:
            triplet = descriptors.sum() * 0.0
        total = self.listwise_weight * listwise + self.triplet_weight * triplet
        return total, {
            "listwise": listwise,
            "triplet": triplet,
            "queries": torch.tensor(float(len(queries)), device=descriptors.device),
            "queries_with_radius_positive_pct": has_positive.float().mean() * 100.0,
            "valid_triplet_queries_pct": valid_triplet.float().mean() * 100.0,
            "mean_positive_count": positive_count.float().mean(),
            "candidate_count": torch.tensor(
                float(candidate_embedding.shape[0]), device=descriptors.device
            ),
            "memory_rows": torch.tensor(float(memory_rows), device=descriptors.device),
        }


def descriptor_preservation_loss(descriptor: Tensor, teacher: Tensor) -> Tensor:
    if descriptor.shape != teacher.shape or descriptor.ndim != 2:
        raise ValueError("student and teacher descriptors must share [batch, features]")
    return (
        1.0 - F.cosine_similarity(descriptor.float(), teacher.float(), dim=1)
    ).mean()


def build_spatial_objective(config: Mapping[str, float]) -> SpatialRetrievalObjective:
    return SpatialRetrievalObjective(
        embedding_temperature=float(config["embedding_temperature"]),
        radius_km=float(config["radius_km"]),
        radius_target_weight=float(config["radius_target_weight"]),
        teacher_target_temperature=float(config["teacher_target_temperature"]),
        soft_distance_temperature_km=float(config["soft_distance_temperature_km"]),
        soft_distance_clip_km=float(config["soft_distance_clip_km"]),
        triplet_margin=float(config["triplet_margin"]),
        negative_minimum_km=float(config["negative_minimum_km"]),
        listwise_weight=float(config["listwise_weight"]),
        triplet_weight=float(config["triplet_weight"]),
    )
