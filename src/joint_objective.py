"""Country-aware joint country-by-region classification objective."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

REGIONS = {"coarse": 6, "a": 8, "b": 12, "c": 18, "fine": 20}
DEFAULT_WEIGHTS = {
    "country": 0.25,
    "coarse": 0.08,
    "a": 0.55 / 3.0,
    "b": 0.55 / 3.0,
    "c": 0.55 / 3.0,
    "fine": 0.12,
}


def flattened_joint_logits(logits: Tensor, regions: int) -> Tensor:
    if logits.ndim != 3 or logits.shape[1:] != (12, regions):
        raise ValueError(f"Expected logits with shape [batch, 12, {regions}]")
    return logits.reshape(logits.shape[0], 12 * regions)


class JointGeoClassificationObjective(nn.Module):
    """Train every spatial head against all wrong-country classes as negatives."""

    def __init__(
        self,
        label_smoothing: float = 0.03,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        if set(self.weights) != set(DEFAULT_WEIGHTS):
            raise ValueError(f"Loss weights must have keys {sorted(DEFAULT_WEIGHTS)}")
        if any(value < 0.0 for value in self.weights.values()):
            raise ValueError("Loss weights cannot be negative")
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("Loss weights must sum to one")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        targets: Mapping[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        country = targets["country"].long()
        if country.ndim != 1 or country.min().item() < 0 or country.max().item() >= 12:
            raise ValueError("country target outside [0, 11]")

        losses: dict[str, Tensor] = {
            "country": F.cross_entropy(
                outputs["country"],
                country,
                label_smoothing=self.label_smoothing,
            )
        }
        for name, regions in REGIONS.items():
            global_target = targets[name].long()
            if global_target.ndim != 1 or global_target.shape != country.shape:
                raise ValueError(f"{name} target must have shape [batch]")
            expected_country = torch.div(global_target, regions, rounding_mode="floor")
            if not torch.equal(expected_country, country):
                raise ValueError(f"{name} target is inconsistent with country target")
            losses[name] = F.cross_entropy(
                flattened_joint_logits(outputs[name], regions),
                global_target,
                label_smoothing=self.label_smoothing,
            )

        total = sum(self.weights[name] * losses[name] for name in self.weights)
        return total, losses
