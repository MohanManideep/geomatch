"""Shared distance metrics and batch helpers for the retrieval pipeline."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor

EARTH_RADIUS_KM = 6371.0088
TARGETS = ("country", "coarse", "a", "b", "c", "fine")


def haversine_km(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("Coordinates must have matching [rows, 2] shapes")
    lat1, lon1 = np.radians(truth[:, 0]), np.radians(truth[:, 1])
    lat2, lon2 = np.radians(prediction[:, 0]), np.radians(prediction[:, 1])
    delta_latitude = lat2 - lat1
    delta_longitude = lon2 - lon1
    haversine = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def distance_summary(distances: np.ndarray) -> dict[str, float]:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1 or len(distances) == 0 or not np.isfinite(distances).all():
        raise ValueError("Distances must be a non-empty finite vector")
    return {
        "median_km": float(np.median(distances)),
        "mean_km": float(np.mean(distances)),
        "p90_km": float(np.percentile(distances, 90)),
        "within_25_pct": float(np.mean(distances <= 25.0) * 100.0),
        "within_100_pct": float(np.mean(distances <= 100.0) * 100.0),
        "within_200_pct": float(np.mean(distances <= 200.0) * 100.0),
        "within_750_pct": float(np.mean(distances <= 750.0) * 100.0),
    }


def move_batch(batch: Mapping[str, Tensor | list[str]], device: torch.device):
    views = tuple(
        batch[name].to(device=device, non_blocking=True)
        for name in ("global_view", "left_view", "right_view")
    )
    targets = {
        name: batch[name].to(device=device, non_blocking=True) for name in TARGETS
    }
    coordinates = batch["coordinates"].to(device=device, non_blocking=True)
    return views, targets, coordinates
