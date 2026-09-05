"""Retrieval model that preserves compatibility with earlier-stage checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from model import (
    EXPECTED_PARAMETER_COUNT,
    PARAMETER_LIMIT,
    GeoCPRegNetY400MF,
    count_trainable_parameters,
)

DESCRIPTOR_FEATURES = 384
LOCAL_FEATURES = 64
LOCAL_GRID_SIZE = 4
LOCAL_TOKENS_PER_VIEW = LOCAL_GRID_SIZE * LOCAL_GRID_SIZE
STAGE4_FEATURES = 440
RETRIEVAL_PROJECTION_PARAMETERS = DESCRIPTOR_FEATURES * DESCRIPTOR_FEATURES


def expected_retrieval_parameter_count(local_features: int) -> int:
    return (
        EXPECTED_PARAMETER_COUNT
        + RETRIEVAL_PROJECTION_PARAMETERS
        + STAGE4_FEATURES * int(local_features)
    )


EXPECTED_RETRIEVAL_PARAMETER_COUNT = expected_retrieval_parameter_count(LOCAL_FEATURES)


class GeoCPRegNetRetrieval(GeoCPRegNetY400MF):
    """Add an identity-initialized metric head without changing the base encoder."""

    def __init__(
        self,
        local_features: int = LOCAL_FEATURES,
        local_grid_size: int = LOCAL_GRID_SIZE,
    ) -> None:
        super().__init__()
        self.local_features = int(local_features)
        self.local_grid_size = int(local_grid_size)
        if not 1 <= self.local_grid_size <= 16:
            raise ValueError(
                "local_grid_size must lie in [1, 16] (stage4 is 16x16 at 512px)"
            )
        if not 1 <= self.local_features <= STAGE4_FEATURES:
            raise ValueError("Local feature dimension must lie in [1, 440]")
        self.retrieval_projection = nn.Linear(
            DESCRIPTOR_FEATURES, DESCRIPTOR_FEATURES, bias=False
        )
        self.local_projection = nn.Linear(
            STAGE4_FEATURES, self.local_features, bias=False
        )
        with torch.no_grad():
            self.retrieval_projection.weight.copy_(
                torch.eye(
                    DESCRIPTOR_FEATURES, dtype=self.retrieval_projection.weight.dtype
                )
            )
            self.local_projection.weight.zero_()
            self.local_projection.weight[:, : self.local_features].copy_(
                torch.eye(self.local_features, dtype=self.local_projection.weight.dtype)
            )
        parameter_count = count_trainable_parameters(self)
        expected_count = expected_retrieval_parameter_count(self.local_features)
        if parameter_count != expected_count:
            raise RuntimeError(
                f"Retrieval model has {parameter_count:,} parameters; "
                f"expected {expected_count:,}"
            )
        if parameter_count > PARAMETER_LIMIT:
            raise RuntimeError("Retrieval model exceeds the parameter limit")

    def forward(
        self, global_view: Tensor, left_view: Tensor, right_view: Tensor
    ) -> dict[str, Tensor]:
        if global_view.ndim != 4 or left_view.ndim != 4 or right_view.ndim != 4:
            raise ValueError("Every input view must have rank four")
        batch = global_view.shape[0]
        global_descriptor, global_tokens = self._encode_view_with_tokens(global_view)
        local_descriptors, local_tokens = self._encode_view_with_tokens(
            torch.cat((left_view, right_view), dim=0)
        )
        left_descriptor, right_descriptor = local_descriptors.split(batch, dim=0)
        left_tokens, right_tokens = local_tokens.split(batch, dim=0)
        descriptor = self.fusion(
            torch.cat((global_descriptor, left_descriptor, right_descriptor), dim=1)
        )
        outputs = {
            "country": self.country_head(descriptor),
            "coarse": self.coarse_head(descriptor).reshape(batch, 12, 6),
            "a": self.a_head(descriptor).reshape(batch, 12, 8),
            "b": self.b_head(descriptor).reshape(batch, 12, 12),
            "c": self.c_head(descriptor).reshape(batch, 12, 18),
            "fine": self.fine_head(descriptor).reshape(batch, 12, 20),
            "descriptor": descriptor,
        }
        outputs["retrieval_descriptor"] = F.normalize(
            self.retrieval_projection(descriptor.float()), dim=1
        )
        outputs["local_tokens"] = torch.cat(
            (global_tokens, left_tokens, right_tokens), dim=1
        )
        return outputs

    def _encode_view_with_tokens(self, image: Tensor) -> tuple[Tensor, Tensor]:
        features = self.stem(image)
        stage1 = self.stages.block1(features)
        stage2 = self.stages.block2(stage1)
        stage3 = self.stages.block3(stage2)
        stage4 = self.stages.block4(stage3)
        pooled = self.pool((stage2, stage3, stage4))
        descriptor = self.view_projection(pooled)
        local_features = F.adaptive_avg_pool2d(
            stage4, (self.local_grid_size, self.local_grid_size)
        )
        local_features = local_features.flatten(2).transpose(1, 2)
        local_tokens = F.normalize(self.local_projection(local_features.float()), dim=2)
        return descriptor, local_tokens


def load_encoder_state_dict(
    model: GeoCPRegNetRetrieval, state_dict: Mapping[str, Tensor]
) -> None:
    """Load a strict encoder state while retaining the identity retrieval head."""

    incompatible = model.load_state_dict(state_dict, strict=False)
    expected_missing = {"retrieval_projection.weight", "local_projection.weight"}
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            f"Unexpected missing encoder checkpoint keys: {incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected encoder checkpoint keys: {incompatible.unexpected_keys}"
        )


def retrieval_identity_error(model: GeoCPRegNetRetrieval, descriptor: Tensor) -> float:
    """Return the maximum identity-initialization error for verification."""

    expected = F.normalize(descriptor.float(), dim=1)
    actual = F.normalize(model.retrieval_projection(descriptor.float()), dim=1)
    return float((expected - actual).abs().max().detach())
