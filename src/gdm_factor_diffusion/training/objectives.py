"""Clean-state and differentiable constraint-guidance objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from gdm_factor_diffusion.diffusion.masking import masked_softmax, validate_state
from gdm_factor_diffusion.graph.batch_adapter import FactorGraphBatch


@dataclass(frozen=True, slots=True)
class ObjectiveTerms:
    total: Tensor
    clean_state: Tensor
    capacity: Tensor
    link: Tensor
    clean_accuracy: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss_total": float(self.total.detach().item()),
            "loss_clean_state": float(self.clean_state.detach().item()),
            "guidance_capacity": float(self.capacity.detach().item()),
            "guidance_link": float(self.link.detach().item()),
            "clean_accuracy": float(self.clean_accuracy.detach().item()),
        }


def clean_state_accuracy(
    logits: Tensor,
    clean_state: Tensor,
    batch: FactorGraphBatch,
) -> Tensor:
    validate_state(clean_state, batch.candidate_mask, batch.service_mask)
    prediction = logits.argmax(dim=-1)
    return (
        prediction[batch.service_mask] == clean_state[batch.service_mask]
    ).float().mean()


def capacity_guidance(
    probability: Tensor,
    batch: FactorGraphBatch,
) -> Tensor:
    """Mean relative expected capacity overload over active device resources."""

    expected_load = torch.einsum(
        "bmd,bmr->bdr",
        probability,
        batch.service_demand,
    )
    relative_load = expected_load / batch.device_capacity.clamp_min(1e-8)
    overload = torch.relu(relative_load - 1.0)
    device_mask = batch.device_node_index >= 0
    active = device_mask.unsqueeze(-1).expand_as(overload)
    return overload[active].mean()


def link_guidance(
    probability: Tensor,
    batch: FactorGraphBatch,
) -> Tensor:
    """Mean probability mass assigned to inadmissible dependency device pairs."""

    source = batch.dependency_index[:, 0].clamp_min(0)
    target = batch.dependency_index[:, 1].clamp_min(0)
    gather_index = source.unsqueeze(-1).expand(-1, -1, probability.shape[-1])
    source_probability = probability.gather(1, gather_index)
    gather_index = target.unsqueeze(-1).expand(-1, -1, probability.shape[-1])
    target_probability = probability.gather(1, gather_index)
    pair_probability = source_probability.unsqueeze(-1) * target_probability.unsqueeze(-2)
    conflict = (
        pair_probability * (~batch.pair_admissible).to(dtype=probability.dtype)
    ).sum(dim=(-1, -2))
    active_conflict = conflict[batch.dependency_mask]
    return (
        active_conflict.mean()
        if active_conflict.numel()
        else probability.new_zeros(())
    )


def compute_objective(
    logits: Tensor,
    clean_state: Tensor,
    batch: FactorGraphBatch,
    *,
    capacity_weight: float,
    link_weight: float,
) -> ObjectiveTerms:
    """Compute the minimal Phase 4A denoiser objective."""

    if capacity_weight < 0 or link_weight < 0:
        raise ValueError("Guidance weights must be nonnegative.")
    validate_state(clean_state, batch.candidate_mask, batch.service_mask)
    clean_loss = F.cross_entropy(
        logits[batch.service_mask],
        clean_state[batch.service_mask],
    )
    probability = masked_softmax(logits, batch.candidate_mask, batch.service_mask)
    capacity = capacity_guidance(probability, batch)
    link = link_guidance(probability, batch)
    total = clean_loss + capacity_weight * capacity + link_weight * link
    return ObjectiveTerms(
        total=total,
        clean_state=clean_loss,
        capacity=capacity,
        link=link,
        clean_accuracy=clean_state_accuracy(logits, clean_state, batch),
    )
