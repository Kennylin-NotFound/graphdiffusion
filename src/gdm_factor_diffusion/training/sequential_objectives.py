"""Teacher-forced objective for sequential conditional placement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from gdm_factor_diffusion.diffusion import (
    PartialPlacementState,
    validate_partial_state,
)
from gdm_factor_diffusion.diffusion.masking import validate_state
from gdm_factor_diffusion.graph import FactorGraphBatch
from gdm_factor_diffusion.inference.masked_decode import build_residual_candidate_mask


@dataclass(frozen=True, slots=True)
class SequentialObjectiveTerms:
    total: Tensor
    reconstruction: Tensor
    next_accuracy: Tensor
    target_count: int

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().item()),
            "loss_next_reconstruction": float(self.reconstruction.detach().item()),
            "next_accuracy": float(self.next_accuracy.detach().item()),
            "target_count": float(self.target_count),
        }


def _validate_order(order: Tensor, batch: FactorGraphBatch) -> Tensor:
    if order.shape != batch.service_mask.shape or order.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("order must be an integer tensor with shape [B, M].")
    normalized = order.to(device=batch.candidate_mask.device, dtype=torch.long)
    for batch_index in range(batch.batch_size):
        active = int(batch.service_mask[batch_index].sum().item())
        expected = set(range(active))
        observed = [int(value) for value in normalized[batch_index, :active].tolist()]
        if set(observed) != expected:
            raise ValueError("order must be a permutation of active services.")
        if (normalized[batch_index, active:] != -1).any():
            raise ValueError("order must use -1 for padded services.")
    return normalized


def build_teacher_forced_prefix(
    clean_state: Tensor,
    order: Tensor,
    step_index: Tensor,
    batch: FactorGraphBatch,
) -> tuple[PartialPlacementState, Tensor, Tensor]:
    """Expose the verified prefix and return the next service to predict."""

    validate_state(clean_state, batch.candidate_mask, batch.service_mask)
    normalized_order = _validate_order(order, batch)
    step = torch.as_tensor(
        step_index,
        dtype=torch.long,
        device=batch.candidate_mask.device,
    )
    if step.ndim == 0:
        step = step.expand(batch.batch_size)
    if step.shape != (batch.batch_size,):
        raise ValueError("step_index must be scalar or have shape [B].")

    active_count = batch.service_mask.sum(dim=1)
    if ((step < 0) | (step >= active_count)).any():
        raise ValueError("step_index must select an active service position.")

    committed = torch.zeros_like(batch.service_mask)
    target_service = torch.empty(
        batch.batch_size,
        dtype=torch.long,
        device=batch.candidate_mask.device,
    )
    for batch_index in range(batch.batch_size):
        current_step = int(step[batch_index].item())
        target_service[batch_index] = normalized_order[batch_index, current_step]
        if current_step:
            prefix = normalized_order[batch_index, :current_step]
            committed[batch_index, prefix] = True

    assignment = clean_state.masked_fill(~committed, -1)
    partial = PartialPlacementState(assignment, committed)
    validate_partial_state(partial, batch.candidate_mask, batch.service_mask)
    step_fraction = step.to(dtype=batch.processing_latency.dtype) / active_count.to(
        dtype=batch.processing_latency.dtype
    ).clamp_min(1)
    return partial, target_service, step_fraction


def compute_sequential_objective(
    logits: Tensor,
    clean_state: Tensor,
    partial_state: PartialPlacementState,
    target_service: Tensor,
    batch: FactorGraphBatch,
    *,
    tolerance: float = 1e-8,
) -> SequentialObjectiveTerms:
    """Apply cross-entropy only to the next service in each prefix."""

    validate_state(clean_state, batch.candidate_mask, batch.service_mask)
    validate_partial_state(partial_state, batch.candidate_mask, batch.service_mask)
    if logits.shape != batch.candidate_mask.shape or not logits.is_floating_point():
        raise ValueError("logits must be floating point with shape [B, M, D].")
    target = torch.as_tensor(
        target_service,
        dtype=torch.long,
        device=batch.candidate_mask.device,
    )
    if target.shape != (batch.batch_size,):
        raise ValueError("target_service must have shape [B].")
    row = torch.arange(batch.batch_size, device=batch.candidate_mask.device)
    if not batch.service_mask[row, target].all():
        raise ValueError("target_service must refer to active services.")
    if partial_state.committed_mask[row, target].any():
        raise ValueError("target_service cannot already be committed.")

    hard_mask = build_residual_candidate_mask(
        batch,
        partial_state,
        tolerance=tolerance,
    )
    target_mask = hard_mask[row, target]
    target_device = clean_state[row, target]
    if not target_mask.any(dim=-1).all():
        raise ValueError("At least one target service has no admissible device.")
    if not target_mask.gather(1, target_device[:, None]).squeeze(1).all():
        raise ValueError("Clean target violates the residual hard mask.")

    target_logits = logits[row, target].masked_fill(~target_mask, -torch.inf)
    reconstruction = F.cross_entropy(target_logits, target_device)
    prediction = target_logits.argmax(dim=-1)
    accuracy = (prediction == target_device).float().mean()
    return SequentialObjectiveTerms(
        total=reconstruction,
        reconstruction=reconstruction,
        next_accuracy=accuracy,
        target_count=batch.batch_size,
    )
