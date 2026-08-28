"""Shared validation and tensor operations for masked categorical variables."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_candidate_mask(
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
) -> Tensor:
    """Validate `[B, M, D]` choices and return the canonical service mask."""

    if candidate_mask.dtype is not torch.bool or candidate_mask.ndim != 3:
        raise ValueError("candidate_mask must be a bool tensor with shape [B, M, D].")
    inferred = candidate_mask.any(dim=-1)
    if service_mask is None:
        return inferred
    if service_mask.dtype is not torch.bool or service_mask.shape != inferred.shape:
        raise ValueError("service_mask must be a bool tensor with shape [B, M].")
    if not torch.equal(service_mask, inferred):
        raise ValueError(
            "Every active service must have candidates and padded services must "
            "have none."
        )
    return service_mask


def validate_state(
    state: Tensor,
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
    *,
    name: str = "state",
) -> Tensor:
    """Require valid candidate indices for active services and `-1` for padding."""

    canonical_service_mask = validate_candidate_mask(candidate_mask, service_mask)
    if state.ndim != 2 or state.shape != canonical_service_mask.shape:
        raise ValueError(f"{name} must have shape [B, M].")
    if state.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError(f"{name} must use an integer dtype.")

    active_state = state[canonical_service_mask]
    if active_state.numel():
        if (active_state < 0).any() or (active_state >= candidate_mask.shape[-1]).any():
            raise ValueError(f"{name} contains an out-of-range active category.")
        selected_is_candidate = candidate_mask.gather(
            -1, state.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)
        if not selected_is_candidate[canonical_service_mask].all():
            raise ValueError(f"{name} selects an incompatible category.")
    if not (state[~canonical_service_mask] == -1).all():
        raise ValueError(f"{name} must use -1 for padded services.")
    return canonical_service_mask


def state_to_one_hot(
    state: Tensor,
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Convert a validated padded state to `[B, M, D]` one-hot form."""

    canonical_service_mask = validate_state(
        state, candidate_mask, service_mask, name="state"
    )
    one_hot = torch.nn.functional.one_hot(
        state.clamp_min(0), num_classes=candidate_mask.shape[-1]
    ).to(dtype=dtype)
    return one_hot * canonical_service_mask.unsqueeze(-1).to(dtype=dtype)


def masked_uniform(
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return a uniform distribution over each active service's candidates."""

    canonical_service_mask = validate_candidate_mask(candidate_mask, service_mask)
    count = candidate_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    probability = candidate_mask.to(dtype=dtype) / count.to(dtype=dtype)
    return probability * canonical_service_mask.unsqueeze(-1).to(dtype=dtype)


def masked_softmax(
    logits: Tensor,
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
) -> Tensor:
    """Softmax only over valid candidates, with zero probability on padding."""

    canonical_service_mask = validate_candidate_mask(candidate_mask, service_mask)
    if logits.shape != candidate_mask.shape or not logits.is_floating_point():
        raise ValueError("logits must be floating point with shape [B, M, D].")
    masked_logits = logits.masked_fill(~candidate_mask, -torch.inf)
    probability = torch.softmax(masked_logits, dim=-1)
    probability = torch.where(
        canonical_service_mask.unsqueeze(-1),
        probability,
        torch.zeros_like(probability),
    )
    return probability.masked_fill(~candidate_mask, 0.0)


def validate_probabilities(
    probability: Tensor,
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
    *,
    tolerance: float = 1e-5,
    name: str = "probability",
) -> Tensor:
    """Validate a padded masked categorical probability tensor."""

    canonical_service_mask = validate_candidate_mask(candidate_mask, service_mask)
    if probability.shape != candidate_mask.shape or not probability.is_floating_point():
        raise ValueError(f"{name} must be floating point with shape [B, M, D].")
    if not torch.isfinite(probability).all() or (probability < -tolerance).any():
        raise ValueError(f"{name} must contain finite nonnegative values.")
    if (probability.masked_select(~candidate_mask).abs() > tolerance).any():
        raise ValueError(f"{name} must be zero outside candidate categories.")
    sums = probability.sum(dim=-1)
    if not torch.allclose(
        sums[canonical_service_mask],
        torch.ones_like(sums[canonical_service_mask]),
        atol=tolerance,
        rtol=tolerance,
    ):
        raise ValueError(f"{name} must sum to one for every active service.")
    if (sums[~canonical_service_mask].abs() > tolerance).any():
        raise ValueError(f"{name} must be zero for padded services.")
    return canonical_service_mask


def normalize_masked(
    value: Tensor,
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
) -> Tensor:
    """Normalize nonnegative masked values without creating padded NaNs."""

    canonical_service_mask = validate_candidate_mask(candidate_mask, service_mask)
    if value.shape != candidate_mask.shape or not value.is_floating_point():
        raise ValueError("value must be floating point with shape [B, M, D].")
    masked = value.masked_fill(~candidate_mask, 0.0)
    denominator = masked.sum(dim=-1, keepdim=True)
    if (denominator[canonical_service_mask] <= 0).any():
        raise ValueError("Active masked rows must have positive mass.")
    normalized = masked / denominator.clamp_min(torch.finfo(masked.dtype).tiny)
    return torch.where(
        canonical_service_mask.unsqueeze(-1),
        normalized,
        torch.zeros_like(normalized),
    )


def sample_categorical(
    probability: Tensor,
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample independent active variables and return `-1` on padding."""

    canonical_service_mask = validate_probabilities(
        probability, candidate_mask, service_mask
    )
    state = torch.full(
        canonical_service_mask.shape,
        -1,
        dtype=torch.long,
        device=probability.device,
    )
    active_probability = probability[canonical_service_mask]
    if active_probability.numel():
        state[canonical_service_mask] = torch.multinomial(
            active_probability,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
    return state
