"""Pure-CNN three-view GeoMatch architecture under the 5M parameter cap."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor, nn
from torchvision.models import regnet_y_400mf

PARAMETER_LIMIT = 5_000_000
EXPECTED_PARAMETER_COUNT = 4_694_295
STAGE_CHANNELS = (104, 208, 440)


class GeoModelOutput(TypedDict):
    country: Tensor
    coarse: Tensor
    a: Tensor
    b: Tensor
    c: Tensor
    fine: Tensor
    descriptor: Tensor


class MultiScaleGeM(nn.Module):
    """Pool three CNN stages with one learned exponent per stage."""

    def __init__(self, stages: int = 3, initial_p: float = 3.0) -> None:
        super().__init__()
        self.raw_p = nn.Parameter(torch.full((stages,), float(initial_p)))

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        if len(features) != len(self.raw_p):
            raise ValueError("Unexpected number of feature stages")
        pooled = []
        for feature, raw_p in zip(features, self.raw_p):
            p = raw_p.clamp(1.0, 6.0)
            value = feature.clamp_min(1e-6).pow(p)
            value = value.mean(dim=(-2, -1)).pow(p.reciprocal())
            pooled.append(value)
        return torch.cat(pooled, dim=1)


class GeoCPRegNetY400MF(nn.Module):
    """Shared RegNet encoder with ordered three-view feature preservation."""

    def __init__(self) -> None:
        super().__init__()
        backbone = regnet_y_400mf(weights=None)
        if backbone.fc.in_features != 440:
            raise RuntimeError(
                f"Unexpected RegNet final width: {backbone.fc.in_features}"
            )
        backbone.fc = nn.Identity()
        self.stem = backbone.stem
        self.stages = backbone.trunk_output

        self.pool = MultiScaleGeM(stages=3)
        self.view_projection = nn.Sequential(
            nn.LayerNorm(sum(STAGE_CHANNELS)),
            nn.Linear(sum(STAGE_CHANNELS), 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.15),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * 256, 384),
            nn.SiLU(),
            nn.LayerNorm(384),
            nn.Dropout(0.20),
        )

        self.country_head = nn.Linear(384, 12)
        self.coarse_head = nn.Linear(384, 12 * 6)
        self.a_head = nn.Linear(384, 12 * 8)
        self.b_head = nn.Linear(384, 12 * 12)
        self.c_head = nn.Linear(384, 12 * 18)
        self.fine_head = nn.Linear(384, 12 * 20)

        parameter_count = count_trainable_parameters(self)
        if parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                f"Parameter count is {parameter_count:,}; "
                f"expected {EXPECTED_PARAMETER_COUNT:,}"
            )
        if parameter_count > PARAMETER_LIMIT:
            raise RuntimeError("Model exceeds the 5,000,000 parameter limit")

    def _encode_view(self, image: Tensor) -> Tensor:
        value = self.stem(image)
        stage1 = self.stages.block1(value)
        stage2 = self.stages.block2(stage1)
        stage3 = self.stages.block3(stage2)
        stage4 = self.stages.block4(stage3)
        pooled = self.pool((stage2, stage3, stage4))
        return self.view_projection(pooled)

    def forward(
        self, global_view: Tensor, left_view: Tensor, right_view: Tensor
    ) -> GeoModelOutput:
        if global_view.ndim != 4 or left_view.ndim != 4 or right_view.ndim != 4:
            raise ValueError(
                "Every input view must have shape [batch, channels, height, width]"
            )
        batch = global_view.shape[0]
        if left_view.shape[0] != batch or right_view.shape[0] != batch:
            raise ValueError("All views must have the same batch size")
        if (
            global_view.shape[1] != 3
            or left_view.shape[1] != 3
            or right_view.shape[1] != 3
        ):
            raise ValueError("Every input view must have exactly three channels")

        global_descriptor = self._encode_view(global_view)
        local_views = torch.cat((left_view, right_view), dim=0)
        local_descriptors = self._encode_view(local_views)
        left_descriptor, right_descriptor = local_descriptors.split(batch, dim=0)
        descriptor = self.fusion(
            torch.cat((global_descriptor, left_descriptor, right_descriptor), dim=1)
        )

        return {
            "country": self.country_head(descriptor),
            "coarse": self.coarse_head(descriptor).reshape(batch, 12, 6),
            "a": self.a_head(descriptor).reshape(batch, 12, 8),
            "b": self.b_head(descriptor).reshape(batch, 12, 12),
            "c": self.c_head(descriptor).reshape(batch, 12, 18),
            "fine": self.fine_head(descriptor).reshape(batch, 12, 20),
            "descriptor": descriptor,
        }


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
