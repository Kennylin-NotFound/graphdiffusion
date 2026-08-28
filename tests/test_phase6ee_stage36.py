from pathlib import Path

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.data.contracts import audit_dataset_config_contract
from gdm_factor_diffusion.experiments.phase6ee_stage36 import (
    evaluate_efficiency_candidates,
)


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]


GATE = {
    "minimum_relative_gap_improvement": 0.05,
    "require_more_paired_wins_than_losses": True,
    "require_raw_feasibility_not_reduced": True,
    "require_final_success_not_reduced": True,
    "maximum_time_ratio_to_deterministic": 1.10,
    "policy": "smallest_passing_proposal_count",
}


def _summary(gap: float, raw: float, total: float) -> dict:
    return {
        "mean_pre_fallback_gap": gap,
        "raw_any_feasibility": raw,
        "final_success_rate": 1.0,
        "mean_sampling_seconds": total - 0.01,
        "mean_total_seconds": total,
    }


def _record(k2_gap: float, k4_gap: float) -> dict:
    return {
        "methods": {
            "masked_deterministic_k1": {
                "pre_fallback_success": True,
                "pre_fallback_gap": 0.04,
            },
            "masked_stochastic_k2": {
                "pre_fallback_success": True,
                "pre_fallback_gap": k2_gap,
            },
            "masked_stochastic_k4": {
                "pre_fallback_success": True,
                "pre_fallback_gap": k4_gap,
            },
        }
    }


def test_selects_smallest_candidate_passing_all_gates() -> None:
    aggregate = {
        "masked_deterministic_k1": _summary(0.04, 0.90, 1.0),
        "masked_stochastic_k2": _summary(0.035, 0.91, 1.08),
        "masked_stochastic_k4": _summary(0.030, 0.92, 1.09),
    }
    result = evaluate_efficiency_candidates(
        aggregate, [_record(0.03, 0.02)], gate=GATE
    )
    assert result["confirmation_authorized"]
    assert result["selected_method"] == "masked_stochastic_k2"


def test_rejects_quality_gain_outside_time_bound() -> None:
    aggregate = {
        "masked_deterministic_k1": _summary(0.04, 0.90, 1.0),
        "masked_stochastic_k2": _summary(0.030, 0.92, 1.11),
        "masked_stochastic_k4": _summary(0.025, 0.93, 1.20),
    }
    result = evaluate_efficiency_candidates(
        aggregate, [_record(0.03, 0.02)], gate=GATE
    )
    assert not result["confirmation_authorized"]
    assert result["selected_method"] is None


def test_rejects_candidate_with_more_paired_losses() -> None:
    aggregate = {
        "masked_deterministic_k1": _summary(0.04, 0.90, 1.0),
        "masked_stochastic_k2": _summary(0.030, 0.92, 1.05),
        "masked_stochastic_k4": _summary(0.025, 0.93, 1.08),
    }
    records = [_record(0.05, 0.05), _record(0.05, 0.05)]
    result = evaluate_efficiency_candidates(aggregate, records, gate=GATE)
    assert not result["confirmation_authorized"]


def test_confirmation_dataset_contract_is_test_only_and_locked() -> None:
    config = load_config(
        IMPLEMENTATION_ROOT
        / "configs"
        / "dataset_phase6e_e_stage36_confirmation.yaml"
    )
    audit_dataset_config_contract(config)
    contract = config["dataset"]["contract"]
    assert contract["expected_instance_count"] == 128
    assert contract["partition_counts"] == {"efficiency_confirmation": 128}
