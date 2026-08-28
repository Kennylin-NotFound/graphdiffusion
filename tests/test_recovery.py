import numpy as np
import pytest

from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    ProposalRecoveryConfig,
    recover_from_proposal,
    solve_from_proposals,
)
from gdm_factor_diffusion.solver import verify_placement


def _link_conflict_proposal() -> np.ndarray:
    return np.asarray([0, 0, 2, 0, 2], dtype=np.int64)


def _proposal_probabilities() -> np.ndarray:
    probability = np.full((5, 3), 0.01, dtype=np.float64)
    proposal = _link_conflict_proposal()
    probability[np.arange(5), proposal] = 0.95
    probability[2, 2] = 0.10
    probability[2, 1] = 0.89
    return probability


def test_recovery_preserves_feasible_proposal_without_changes() -> None:
    instance = create_toy_instance()
    placement = np.asarray([0, 1, 1, 0, 1], dtype=np.int64)

    result = recover_from_proposal(instance, placement)

    assert result.success
    assert result.released_services == ()
    assert result.completion_order == ()
    assert np.array_equal(result.placement, placement)


def test_recovery_releases_conflicting_endpoint_and_uses_model_scores() -> None:
    instance = create_toy_instance()

    result = recover_from_proposal(
        instance,
        _link_conflict_proposal(),
        model_probability=_proposal_probabilities(),
        config=ProposalRecoveryConfig(max_released_services=1),
    )

    assert result.success
    assert result.released_services == (2,)
    assert result.completion_order == (2,)
    assert result.placement is not None
    assert result.placement[2] == 1
    assert verify_placement(instance, result.placement).feasible


def test_recovery_fails_cleanly_when_release_budget_is_exceeded() -> None:
    result = recover_from_proposal(
        create_toy_instance(),
        _link_conflict_proposal(),
        model_probability=_proposal_probabilities(),
        config=ProposalRecoveryConfig(max_released_services=0),
    )

    assert not result.success
    assert result.placement is None
    assert result.failure_reason == "release_budget_exceeded"


def test_solver_invokes_recovery_only_when_raw_candidate_set_is_empty() -> None:
    instance = create_toy_instance()
    proposal = _link_conflict_proposal()[None, :]
    probability = _proposal_probabilities()[None, :, :]
    config = InferenceConfig(
        num_samples=1,
        sample_batch_size=1,
        enable_repair=False,
        enable_recovery=True,
        enable_fallback=False,
        recovery_candidate_limit=1,
        recovery_max_released_services=1,
    )

    recovered = solve_from_proposals(
        instance,
        proposal,
        model_probabilities=probability,
        config=config,
        proposal_method="recovery-unit",
    )
    assert recovered.success
    assert recovered.source == "recovery"
    assert recovered.metrics["raw_any_feasible"] is False
    assert recovered.metrics["recovery_invoked"] is True
    assert recovered.metrics["recovery_attempts"] == 1
    assert recovered.metrics["recovery_successes"] == 1
    assert recovered.metrics["fallback_invoked"] is False

    raw_feasible = solve_from_proposals(
        instance,
        np.asarray([[0, 1, 1, 0, 1]], dtype=np.int64),
        config=config,
        proposal_method="recovery-unit",
    )
    assert raw_feasible.success
    assert raw_feasible.source == "raw"
    assert raw_feasible.metrics["recovery_invoked"] is False
    assert raw_feasible.metrics["recovery_attempts"] == 0


def test_recovery_cannot_be_mixed_with_legacy_postprocessors() -> None:
    with pytest.raises(ValueError, match="legacy repair"):
        InferenceConfig(
            enable_recovery=True,
            enable_repair=True,
            enable_fallback=False,
        ).validate()
    with pytest.raises(ValueError, match="legacy fallback"):
        InferenceConfig(
            enable_recovery=True,
            enable_repair=False,
            enable_fallback=True,
        ).validate()

