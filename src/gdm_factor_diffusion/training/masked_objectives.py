"""Reconstruction objective for absorbing-MASK conditional placement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from gdm_factor_diffusion.diffusion import (
    PartialPlacementState,
    hidden_service_mask,
    validate_state,
)
from gdm_factor_diffusion.graph import FactorGraphBatch


@dataclass(frozen=True, slots=True)
class MaskedObjectiveTerms:
    total: Tensor
    reconstruction: Tensor
    hidden_accuracy: Tensor
    hidden_count: int

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().item()),
            "loss_masked_reconstruction": float(self.reconstruction.detach().item()),
            "hidden_accuracy": float(self.hidden_accuracy.detach().item()),
            "hidden_count": float(self.hidden_count),
        }


def compute_masked_objective(
    logits: Tensor,
    clean_state: Tensor,
    partial_state: PartialPlacementState,
    batch: FactorGraphBatch,
) -> MaskedObjectiveTerms:
    """Apply cross-entropy only to active assignments hidden by corruption."""

    validate_state(clean_state, batch.candidate_mask, batch.service_mask)
    if logits.shape != batch.candidate_mask.shape or not logits.is_floating_point():
        raise ValueError("logits must be floating point with shape [B, M, D].")
    hidden = hidden_service_mask(
        partial_state,
        batch.candidate_mask,
        batch.service_mask,
    )
    hidden_count = int(hidden.sum().item())
    if hidden_count < 1:
        raise ValueError("Masked reconstruction requires at least one hidden service.")
    reconstruction = F.cross_entropy(logits[hidden], clean_state[hidden])
    prediction = logits.argmax(dim=-1)
    accuracy = (prediction[hidden] == clean_state[hidden]).float().mean()
    return MaskedObjectiveTerms(
        total=reconstruction,
        reconstruction=reconstruction,
        hidden_accuracy=accuracy,
        hidden_count=hidden_count,
    )
