from gdm_factor_diffusion.experiments.phase6ee_stage3_pilot import (
    evaluate_stage3_gates,
)


def _aggregate(total: float, gap: float, raw: float) -> dict:
    return {
        "final_success_rate": 1.0,
        "mean_pre_fallback_gap": gap,
        "raw_any_feasibility": raw,
        "mean_total_seconds": total,
    }


def _records(stochastic_wins: bool) -> list[dict]:
    det = 0.04 if stochastic_wins else 0.02
    sto = 0.02 if stochastic_wins else 0.04
    return [
        {
            "methods": {
                "masked_deterministic_k1": {
                    "pre_fallback_success": True,
                    "pre_fallback_gap": det,
                },
                "masked_stochastic_k8": {
                    "pre_fallback_success": True,
                    "pre_fallback_gap": sto,
                },
            }
        }
    ]


GATES = {
    "partial_conditioning": {
        "minimum_relative_gap_improvement": 0.10,
        "minimum_raw_any_percentage_point_improvement": 5.0,
        "maximum_time_ratio_to_direct": 1.50,
    },
    "diffusion_specific": {
        "minimum_relative_gap_improvement": 0.05,
        "maximum_time_ratio_to_deterministic": 1.10,
    },
}


def test_gate_outcome_a_requires_both_gates() -> None:
    aggregate = {
        "direct_k32": _aggregate(1.0, 0.08, 0.50),
        "masked_deterministic_k1": _aggregate(1.2, 0.04, 0.70),
        "masked_stochastic_k8": _aggregate(1.3, 0.02, 0.75),
    }
    result = evaluate_stage3_gates(
        aggregate,
        _records(True),
        direct_id="direct_k32",
        stochastic_id="masked_stochastic_k8",
        gates=GATES,
    )
    assert result["gate_r3b"]["passed"]
    assert result["gate_r3c"]["passed"]
    assert result["outcome"] == "A"


def test_gate_outcome_b_when_stochastic_time_is_too_high() -> None:
    aggregate = {
        "direct_k32": _aggregate(1.0, 0.08, 0.50),
        "masked_deterministic_k1": _aggregate(1.2, 0.04, 0.70),
        "masked_stochastic_k8": _aggregate(1.8, 0.02, 0.75),
    }
    result = evaluate_stage3_gates(
        aggregate,
        _records(True),
        direct_id="direct_k32",
        stochastic_id="masked_stochastic_k8",
        gates=GATES,
    )
    assert result["gate_r3b"]["passed"]
    assert not result["gate_r3c"]["passed"]
    assert result["outcome"] == "B"


def test_gate_outcome_c_when_conditioning_does_not_help() -> None:
    aggregate = {
        "direct_k32": _aggregate(1.0, 0.04, 0.70),
        "masked_deterministic_k1": _aggregate(1.2, 0.05, 0.70),
        "masked_stochastic_k8": _aggregate(1.25, 0.04, 0.72),
    }
    result = evaluate_stage3_gates(
        aggregate,
        _records(False),
        direct_id="direct_k32",
        stochastic_id="masked_stochastic_k8",
        gates=GATES,
    )
    assert not result["gate_r3b"]["passed"]
    assert result["outcome"] == "C"
