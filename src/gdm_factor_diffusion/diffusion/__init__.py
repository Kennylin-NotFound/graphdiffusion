"""Masked categorical diffusion for service-to-device placements."""

from .categorical_diffusion import (
    build_reverse_timestep_grid,
    model_posterior,
    model_posterior_between,
    p_sample,
    p_sample_to,
    q_probabilities,
    q_posterior,
    q_sample,
    reverse_sample_loop,
    sample_prior,
)
from .categorical_schedule import CategoricalSchedule
from .masking import (
    masked_softmax,
    masked_uniform,
    sample_categorical,
    state_to_one_hot,
    validate_candidate_mask,
    validate_probabilities,
    validate_state,
)
from .partial_mask import (
    AbsorbingMaskSchedule,
    PartialPlacementState,
    all_masked_state,
    corrupt_with_absorbing_mask,
    hidden_service_mask,
    validate_partial_state,
)

__all__ = [
    "CategoricalSchedule",
    "AbsorbingMaskSchedule",
    "PartialPlacementState",
    "all_masked_state",
    "build_reverse_timestep_grid",
    "corrupt_with_absorbing_mask",
    "hidden_service_mask",
    "masked_softmax",
    "masked_uniform",
    "model_posterior",
    "model_posterior_between",
    "p_sample",
    "p_sample_to",
    "q_posterior",
    "q_probabilities",
    "q_sample",
    "reverse_sample_loop",
    "sample_categorical",
    "sample_prior",
    "state_to_one_hot",
    "validate_candidate_mask",
    "validate_probabilities",
    "validate_partial_state",
    "validate_state",
]
