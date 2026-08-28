"""Absorbing-MASK states for partial service-placement assignments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .masking import validate_candidate_mask, validate_state


@dataclass(frozen=True, slots=True)
class PartialPlacementState:
    """A padded placement whose committed assignments are explicitly visible."""

    assignment: Tensor
    committed_mask: Tensor

    def to(self, device: torch.device | str) -> "PartialPlacementState":
        return PartialPlacementState(
            assignment=self.assignment.to(device),
            committed_mask=self.committed_mask.to(device),
        )


@dataclass(frozen=True, slots=True)
class AbsorbingMaskSchedule:
    """Monotone schedule with no masks at step zero and all masks at step T."""

    num_steps: int = 8
    power: float = 1.0

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive.")
        if self.power <= 0:
            raise ValueError("power must be positive.")

    def mask_probability(
        self,
        timestep: int | Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        value = torch.as_tensor(timestep, device=device, dtype=dtype)
        if (value < 0).any() or (value > self.num_steps).any():
            raise ValueError("timestep is outside [0, num_steps].")
        return (value / float(self.num_steps)).pow(self.power)


def validate_partial_state(
    state: PartialPlacementState,
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
    *,
    name: str = "partial_state",
) -> Tensor:
    """Validate explicit committed, uncommitted, and padded placement entries."""

    canonical = validate_candidate_mask(candidate_mask, service_mask)
    assignment = state.assignment
    committed = state.committed_mask
    if assignment.shape != canonical.shape or assignment.ndim != 2:
        raise ValueError(f"{name}.assignment must have shape [B, M].")
    if assignment.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError(f"{name}.assignment must use an integer dtype.")
    if committed.dtype is not torch.bool or committed.shape != canonical.shape:
        raise ValueError(f"{name}.committed_mask must be bool with shape [B, M].")
    if (committed & ~canonical).any():
        raise ValueError(f"{name} cannot commit padded services.")
    if not (assignment[~committed] == -1).all():
        raise ValueError(f"{name} must use -1 for every uncommitted service.")

    if committed.any():
        selected = assignment[committed]
        if (selected < 0).any() or (selected >= candidate_mask.shape[-1]).any():
            raise ValueError(f"{name} contains an out-of-range committed category.")
        selected_is_candidate = candidate_mask.gather(
            -1, assignment.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)
        if not selected_is_candidate[committed].all():
            raise ValueError(f"{name} commits an incompatible category.")
    return canonical


def all_masked_state(
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
) -> PartialPlacementState:
    """Create the absorbing all-MASK state without overloading padding semantics."""

    canonical = validate_candidate_mask(candidate_mask, service_mask)
    return PartialPlacementState(
        assignment=torch.full(
            canonical.shape,
            -1,
            dtype=torch.long,
            device=candidate_mask.device,
        ),
        committed_mask=torch.zeros_like(canonical),
    )


def _normalize_timestep(
    timestep: int | Tensor,
    batch_size: int,
    schedule: AbsorbingMaskSchedule,
    device: torch.device,
) -> Tensor:
    value = torch.as_tensor(timestep, dtype=torch.long, device=device)
    if value.ndim == 0:
        value = value.expand(batch_size)
    if value.shape != (batch_size,):
        raise ValueError("timestep must be scalar or have shape [B].")
    if (value < 0).any() or (value > schedule.num_steps).any():
        raise ValueError("timestep is outside [0, num_steps].")
    return value


def corrupt_with_absorbing_mask(
    clean_state: Tensor,
    timestep: int | Tensor,
    candidate_mask: Tensor,
    schedule: AbsorbingMaskSchedule,
    service_mask: Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> PartialPlacementState:
    """Independently hide clean assignments according to the absorbing schedule."""

    canonical = validate_state(clean_state, candidate_mask, service_mask)
    steps = _normalize_timestep(
        timestep,
        clean_state.shape[0],
        schedule,
        clean_state.device,
    )
    probability = schedule.mask_probability(
        steps,
        device=clean_state.device,
        dtype=torch.float32,
    )
    random_value = torch.rand(
        canonical.shape,
        device=clean_state.device,
        generator=generator,
    )
    hidden = (random_value < probability[:, None]) & canonical
    hidden = torch.where(
        (steps == schedule.num_steps)[:, None], canonical, hidden
    )
    hidden = torch.where((steps == 0)[:, None], torch.zeros_like(hidden), hidden)

    positive = steps > 0
    missing_hidden = positive & ~hidden.any(dim=1)
    for batch_index in torch.nonzero(missing_hidden, as_tuple=False).flatten():
        choices = torch.nonzero(canonical[batch_index], as_tuple=False).flatten()
        selected = torch.randint(
            choices.numel(),
            (1,),
            device=clean_state.device,
            generator=generator,
        )
        hidden[batch_index, choices[selected]] = True

    committed = canonical & ~hidden
    assignment = clean_state.clone().masked_fill(~committed, -1)
    state = PartialPlacementState(assignment, committed)
    validate_partial_state(state, candidate_mask, canonical)
    return state


def hidden_service_mask(
    state: PartialPlacementState,
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
) -> Tensor:
    """Return active services whose clean assignment is hidden."""

    canonical = validate_partial_state(state, candidate_mask, service_mask)
    return canonical & ~state.committed_mask
