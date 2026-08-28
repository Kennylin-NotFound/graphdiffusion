from copy import deepcopy

from gdm_factor_diffusion.data.contracts import audit_dataset_config_contract
from gdm_factor_diffusion.experiments.phase6ee_stage38 import (
    evaluate_stage38_gate,
)


ACCEPTANCE = {
    "minimum_time_ratio": 0.90,
    "maximum_time_ratio": 1.10,
    "minimum_aggregate_relative_pre_fallback_gap_improvement": 0.10,
    "require_positive_pre_fallback_improvement_each_seed": True,
    "require_pre_fallback_success_not_reduced": True,
    "require_raw_feasibility_not_reduced": True,
    "require_final_gap_not_worse": True,
    "require_final_success_not_reduced": True,
    "require_more_instance_wins_than_losses": True,
    "maximum_instance_sign_test_pvalue": 0.05,
}


def _method(pre_gap: float, total: float, raw: bool = True) -> dict:
    return {
        "success": True,
        "gap_to_pool_best": pre_gap,
        "pre_fallback_success": True,
        "pre_fallback_gap": pre_gap,
        "raw_any_feasible": raw,
        "fallback_invoked": False,
        "sampling_seconds": total * 0.8,
        "total_seconds": total,
    }


def _records(diffusion_gap: float = 0.02, diffusion_time: float = 1.05):
    records = []
    for seed in (1, 2, 3):
        for instance in range(16):
            records.append(
                {
                    "training_seed": seed,
                    "instance_id": f"i{instance:02d}",
                    "methods": {
                        "direct_k64": _method(0.04, 1.0, raw=False),
                        "masked_deterministic_k1": _method(0.03, 0.7),
                        "masked_diffusion_k8": _method(
                            diffusion_gap, diffusion_time
                        ),
                    },
                }
            )
    return records


def test_stage38_gate_accepts_replicated_matched_time_advantage() -> None:
    result = evaluate_stage38_gate(_records(), [1, 2, 3], ACCEPTANCE)
    assert result["passed"]
    assert result["instance_paired_wins"] == 16
    assert result["instance_sign_test_pvalue"] < 0.05


def test_stage38_gate_rejects_one_seed_runtime_drift() -> None:
    records = _records()
    for row in records:
        if row["training_seed"] == 3:
            row["methods"]["masked_diffusion_k8"]["total_seconds"] = 1.11
    result = evaluate_stage38_gate(records, [1, 2, 3], ACCEPTANCE)
    assert not result["passed"]
    assert not result["checks"]["time_comparable_each_seed"]


def test_stage38_gate_rejects_no_quality_advantage() -> None:
    result = evaluate_stage38_gate(
        _records(diffusion_gap=0.05), [1, 2, 3], ACCEPTANCE
    )
    assert not result["passed"]


def test_stage38_dataset_contract_is_test_only() -> None:
    config = {
        "dataset": {
            "contract": {
                "family": "stage38_sealed_confirmation",
                "expected_instance_count": 128,
                "partition_counts": {"sealed_test_id": 128},
                "labeling": {
                    "target_size": 16,
                    "total_time_limit_seconds": 120,
                    "threads": 1,
                },
            },
            "partitions": {
                "sealed_test_id": {
                    "count": 128,
                    "role": "test",
                    "regime": "in_distribution",
                }
            },
        }
    }
    audit_dataset_config_contract(config)
    invalid = deepcopy(config)
    invalid["dataset"]["partitions"]["sealed_test_id"]["role"] = "validation"
    try:
        audit_dataset_config_contract(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("A non-test sealed partition must be rejected.")
