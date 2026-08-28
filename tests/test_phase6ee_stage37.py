from gdm_factor_diffusion.experiments.phase6ee_stage37 import (
    evaluate_matched_time_gate,
)


GATE = {
    "minimum_time_ratio": 0.90,
    "maximum_time_ratio": 1.10,
    "minimum_relative_pre_fallback_gap_improvement": 0.10,
    "require_more_paired_wins_than_losses": True,
    "require_raw_feasibility_not_reduced": True,
    "require_final_gap_not_worse": True,
    "require_final_success_not_reduced": True,
}


def _summary(total, pre, final, raw):
    return {
        "mean_total_seconds": total,
        "mean_pre_fallback_gap": pre,
        "mean_gap_to_pool_best": final,
        "raw_any_feasibility": raw,
        "final_success_rate": 1.0,
    }


def _records(diffusion_gap, direct_gap):
    return [
        {
            "methods": {
                "masked_diffusion_k8": {
                    "pre_fallback_success": True,
                    "pre_fallback_gap": diffusion_gap,
                },
                "direct_k64": {
                    "pre_fallback_success": True,
                    "pre_fallback_gap": direct_gap,
                },
            }
        }
    ]


def test_matched_time_gate_accepts_quality_advantage() -> None:
    aggregate = {
        "direct_k64": _summary(1.0, 0.04, 0.02, 0.60),
        "masked_diffusion_k8": _summary(1.05, 0.02, 0.01, 0.90),
    }
    result = evaluate_matched_time_gate(
        aggregate, _records(0.02, 0.04), selected_direct="direct_k64", gate=GATE
    )
    assert result["passed"]


def test_matched_time_gate_rejects_unmatched_runtime() -> None:
    aggregate = {
        "direct_k64": _summary(1.0, 0.04, 0.02, 0.60),
        "masked_diffusion_k8": _summary(1.11, 0.02, 0.01, 0.90),
    }
    result = evaluate_matched_time_gate(
        aggregate, _records(0.02, 0.04), selected_direct="direct_k64", gate=GATE
    )
    assert not result["passed"]


def test_matched_time_gate_rejects_direct_quality_win() -> None:
    aggregate = {
        "direct_k64": _summary(1.0, 0.02, 0.01, 0.90),
        "masked_diffusion_k8": _summary(1.02, 0.03, 0.02, 0.90),
    }
    result = evaluate_matched_time_gate(
        aggregate, _records(0.03, 0.02), selected_direct="direct_k64", gate=GATE
    )
    assert not result["passed"]

