"""Analytical forward and reverse operations for masked categorical diffusion."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from .categorical_schedule import CategoricalSchedule
from .masking import (
    masked_softmax,
    masked_uniform,
    normalize_masked,
    sample_categorical,
    state_to_one_hot,
    validate_candidate_mask,
    validate_probabilities,
    validate_state,
)


def _batch_scalar(value: Tensor, target_ndim: int) -> Tensor:
    return value.reshape(value.shape[0], *([1] * (target_ndim - 1)))


def q_probabilities(
    clean_state: Tensor,
    timestep: int | Tensor,
    candidate_mask: Tensor,
    schedule: CategoricalSchedule,
    service_mask: Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute the cumulative forward marginal `q(z_t | z_0)`."""

    canonical_service_mask = validate_state(
        clean_state, candidate_mask, service_mask, name="clean_state"
    )
    timesteps = schedule.normalize_timesteps(
        timestep,
        batch_size=clean_state.shape[0],
        device=clean_state.device,
    )
    retention = _batch_scalar(
        schedule.alpha_bar_at(timesteps, dtype=dtype), candidate_mask.ndim
    )
    clean_one_hot = state_to_one_hot(
        clean_state,
        candidate_mask,
        canonical_service_mask,
        dtype=dtype,
    )
    uniform = masked_uniform(
        candidate_mask,
        canonical_service_mask,
        dtype=dtype,
    )
    return retention * clean_one_hot + (1.0 - retention) * uniform


def q_sample(
    clean_state: Tensor,
    timestep: int | Tensor,
    candidate_mask: Tensor,
    schedule: CategoricalSchedule,
    service_mask: Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample `z_t` directly from the cumulative forward marginal."""

    probability = q_probabilities(
        clean_state,
        timestep,
        candidate_mask,
        schedule,
        service_mask,
    )
    return sample_categorical(
        probability,
        candidate_mask,
        service_mask,
        generator=generator,
    )


def q_posterior(
    noisy_state: Tensor,
    clean_state: Tensor,
    timestep: int | Tensor,
    candidate_mask: Tensor,
    schedule: CategoricalSchedule,
    service_mask: Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute the exact per-variable posterior `q(z_{t-1} | z_t, z_0)`."""

    canonical_service_mask = validate_state(
        clean_state, candidate_mask, service_mask, name="clean_state"
    )
    validate_state(
        noisy_state, candidate_mask, canonical_service_mask, name="noisy_state"
    )
    timesteps = schedule.normalize_timesteps(
        timestep,
        batch_size=clean_state.shape[0],
        device=clean_state.device,
    )
    previous_retention = _batch_scalar(
        schedule.alpha_bar_at(timesteps - 1, dtype=dtype),
        candidate_mask.ndim,
    )
    beta = _batch_scalar(
        schedule.beta_at(timesteps, dtype=dtype),
        candidate_mask.ndim,
    )
    uniform = masked_uniform(
        candidate_mask,
        canonical_service_mask,
        dtype=dtype,
    )
    clean_one_hot = state_to_one_hot(
        clean_state,
        candidate_mask,
        canonical_service_mask,
        dtype=dtype,
    )
    noisy_one_hot = state_to_one_hot(
        noisy_state,
        candidate_mask,
        canonical_service_mask,
        dtype=dtype,
    )
    previous_marginal = (
        previous_retention * clean_one_hot
        + (1.0 - previous_retention) * uniform
    )
    transition_likelihood = beta * uniform + (1.0 - beta) * noisy_one_hot
    return normalize_masked(
        previous_marginal * transition_likelihood,
        candidate_mask,
        canonical_service_mask,
    )


def model_posterior(
    clean_probability: Tensor,
    noisy_state: Tensor,
    timestep: int | Tensor,
    candidate_mask: Tensor,
    schedule: CategoricalSchedule,
    service_mask: Tensor | None = None,
) -> Tensor:
    """Mix exact posteriors using predicted `p_theta(z_0 | z_t, H)`."""

    timesteps = schedule.normalize_timesteps(
        timestep,
        batch_size=noisy_state.shape[0],
        device=noisy_state.device,
    )
    return model_posterior_between(
        clean_probability,
        noisy_state,
        timesteps,
        timesteps - 1,
        candidate_mask,
        schedule,
        service_mask,
    )


def model_posterior_between(
    clean_probability: Tensor,
    noisy_state: Tensor,
    timestep: int | Tensor,
    previous_timestep: int | Tensor,
    candidate_mask: Tensor,
    schedule: CategoricalSchedule,
    service_mask: Tensor | None = None,
) -> Tensor:
    """Mix exact `q(z_s | z_t, z_0)` posteriors for any `0 <= s < t`."""

    canonical_service_mask = validate_probabilities(
        clean_probability,
        candidate_mask,
        service_mask,
        name="clean_probability",
    )
    validate_state(
        noisy_state, candidate_mask, canonical_service_mask, name="noisy_state"
    )
    timesteps = schedule.normalize_timesteps(
        timestep,
        batch_size=noisy_state.shape[0],
        device=noisy_state.device,
    )
    previous_timesteps = schedule.normalize_timesteps(
        previous_timestep,
        batch_size=noisy_state.shape[0],
        device=noisy_state.device,
        allow_zero=True,
    )
    if (previous_timesteps >= timesteps).any():
        raise ValueError("previous_timestep must be strictly smaller than timestep.")
    dtype = clean_probability.dtype
    previous_retention = _batch_scalar(
        schedule.alpha_bar_at(previous_timesteps, dtype=dtype),
        candidate_mask.ndim,
    ).unsqueeze(-1)
    current_retention = schedule.alpha_bar_at(timesteps, dtype=dtype)
    previous_retention_flat = schedule.alpha_bar_at(
        previous_timesteps, dtype=dtype
    )
    jump_beta = _batch_scalar(
        1.0 - current_retention / previous_retention_flat,
        candidate_mask.ndim,
    )
    uniform = masked_uniform(
        candidate_mask,
        canonical_service_mask,
        dtype=dtype,
    )
    noisy_one_hot = state_to_one_hot(
        noisy_state,
        candidate_mask,
        canonical_service_mask,
        dtype=dtype,
    )

    categories = candidate_mask.shape[-1]
    identity = torch.eye(categories, dtype=dtype, device=candidate_mask.device)
    previous_given_clean = (
        previous_retention * identity.reshape(1, 1, categories, categories)
        + (1.0 - previous_retention) * uniform.unsqueeze(-2)
    )
    valid_clean_previous = (
        candidate_mask.unsqueeze(-1) & candidate_mask.unsqueeze(-2)
    )
    previous_given_clean = previous_given_clean.masked_fill(
        ~valid_clean_previous, 0.0
    )

    transition_likelihood = (
        jump_beta * uniform + (1.0 - jump_beta) * noisy_one_hot
    ).unsqueeze(-2)
    unnormalized = previous_given_clean * transition_likelihood
    denominator = unnormalized.sum(dim=-1, keepdim=True)
    exact_given_clean = unnormalized / denominator.clamp_min(
        torch.finfo(dtype).tiny
    )
    exact_given_clean = exact_given_clean.masked_fill(~valid_clean_previous, 0.0)

    mixed = (clean_probability.unsqueeze(-1) * exact_given_clean).sum(dim=-2)
    return normalize_masked(
        mixed,
        candidate_mask,
        canonical_service_mask,
    )


def p_sample(
    clean_logits: Tensor,
    noisy_state: Tensor,
    timestep: int | Tensor,
    candidate_mask: Tensor,
    schedule: CategoricalSchedule,
    service_mask: Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one reverse step from denoiser clean-state logits."""

    timesteps = schedule.normalize_timesteps(
        timestep,
        batch_size=noisy_state.shape[0],
        device=noisy_state.device,
    )
    return p_sample_to(
        clean_logits,
        noisy_state,
        timesteps,
        timesteps - 1,
        candidate_mask,
        schedule,
        service_mask,
        generator=generator,
    )


def p_sample_to(
    clean_logits: Tensor,
    noisy_state: Tensor,
    timestep: int | Tensor,
    previous_timestep: int | Tensor,
    candidate_mask: Tensor,
    schedule: CategoricalSchedule,
    service_mask: Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample an analytical reverse transition from timestep `t` to `s < t`."""

    clean_probability = masked_softmax(clean_logits, candidate_mask, service_mask)
    posterior = model_posterior_between(
        clean_probability,
        noisy_state,
        timestep,
        previous_timestep,
        candidate_mask,
        schedule,
        service_mask,
    )
    return sample_categorical(
        posterior,
        candidate_mask,
        service_mask,
        generator=generator,
    )


def sample_prior(
    candidate_mask: Tensor,
    service_mask: Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample the factorized uniform terminal prior over compatible devices."""

    canonical_service_mask = validate_candidate_mask(candidate_mask, service_mask)
    probability = masked_uniform(candidate_mask, canonical_service_mask)
    return sample_categorical(
        probability,
        candidate_mask,
        canonical_service_mask,
        generator=generator,
    )


def reverse_sample_loop(
    clean_logits_fn: Callable[[Tensor, Tensor], Tensor],
    candidate_mask: Tensor,
    schedule: CategoricalSchedule,
    service_mask: Tensor | None = None,
    *,
    initial_state: Tensor | None = None,
    timesteps: tuple[int, ...] | None = None,
    generator: torch.Generator | None = None,
    step_callback: Callable[[int, int, Tensor, Tensor, Tensor], None] | None = None,
) -> Tensor:
    """Run adjacent or analytical skipped reverse transitions."""

    canonical_service_mask = validate_candidate_mask(candidate_mask, service_mask)
    state = (
        sample_prior(
            candidate_mask,
            canonical_service_mask,
            generator=generator,
        )
        if initial_state is None
        else initial_state
    )
    validate_state(state, candidate_mask, canonical_service_mask, name="initial_state")
    batch_size = candidate_mask.shape[0]
    grid = (
        tuple(range(schedule.num_steps, -1, -1))
        if timesteps is None
        else tuple(int(step) for step in timesteps)
    )
    if (
        len(grid) < 2
        or grid[0] != schedule.num_steps
        or grid[-1] != 0
        or any(current <= previous for current, previous in zip(grid, grid[1:]))
    ):
        raise ValueError(
            "timesteps must be strictly descending from schedule.num_steps to 0."
        )
    for step, previous_step in zip(grid, grid[1:]):
        timestep = torch.full(
            (batch_size,),
            step,
            dtype=torch.long,
            device=candidate_mask.device,
        )
        clean_logits = clean_logits_fn(state, timestep)
        previous_state = p_sample_to(
            clean_logits,
            state,
            timestep,
            previous_step,
            candidate_mask,
            schedule,
            canonical_service_mask,
            generator=generator,
        )
        if step_callback is not None:
            step_callback(
                step,
                previous_step,
                state.detach().clone(),
                clean_logits.detach().clone(),
                previous_state.detach().clone(),
            )
        state = previous_state
    return state


def build_reverse_timestep_grid(
    num_steps: int,
    num_transitions: int,
) -> tuple[int, ...]:
    """Build a deterministic approximately uniform grid from `T` through `0`."""

    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if num_transitions < 1 or num_transitions > num_steps:
        raise ValueError("num_transitions must lie in [1, num_steps].")
    return tuple(
        round(num_steps * (num_transitions - index) / num_transitions)
        for index in range(num_transitions + 1)
    )
