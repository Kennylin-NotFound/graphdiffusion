import json
from pathlib import Path

import pytest

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.experiments import (
    ExperimentManifest,
    MethodSpec,
    aggregate_run_directories,
    aggregate_phase6ee_stage1_records,
    audit_training_run,
    evaluate_experiment,
    export_run_figures,
    export_training_curves,
    select_time_matched_budget,
    verify_checkpoint_freeze,
    verify_phase6d_c_evidence,
    verify_phase6e_a_lock,
    verify_phase6eb_lock,
    verify_phase6eb_final_evidence,
)
from gdm_factor_diffusion.experiments.sealed_campaign import verify_sealed_campaign_lock
from gdm_factor_diffusion.experiments.aggregation import quality_fingerprint
from gdm_factor_diffusion.experiments.aggregation import pairwise_outcomes
from gdm_factor_diffusion.experiments.scale_probe import ScaleProbeConfig


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
FROZEN_DATASET = "artifacts/datasets/phase5b-pilot"


def _manifest(output_root: Path) -> ExperimentManifest:
    return ExperimentManifest(
        name="experiment-unit",
        dataset_root=FROZEN_DATASET,
        partitions=("test_id",),
        methods=(
            MethodSpec(
                method_id="random",
                kind="random_hybrid",
                proposal_group="random-shared",
                inference={"num_samples": 2, "sample_batch_size": 2},
            ),
            MethodSpec(method_id="fallback", kind="fallback_only"),
        ),
        seed=17,
        device="cpu",
        output_root=str(output_root),
        instance_limit=1,
    )


def test_experiment_manifest_rejects_unknown_claim_method() -> None:
    manifest = _manifest(Path("unused"))
    payload = manifest.to_dict()
    payload["methods"][0]["kind"] = "unknown"
    with pytest.raises(ValueError):
        MethodSpec(**payload["methods"][0]).validate()


def test_shared_evaluator_writes_reproducible_quality_artifacts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    first = evaluate_experiment(manifest, implementation_root=IMPLEMENTATION_ROOT)
    second = evaluate_experiment(manifest, implementation_root=IMPLEMENTATION_ROOT)
    first_summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    second_summary = json.loads((second / "summary.json").read_text(encoding="utf-8"))

    assert first_summary["quality_fingerprint"] == second_summary["quality_fingerprint"]
    assert first_summary["records"] == 2
    assert first_summary["all_final_outputs_verified"]
    assert first_summary["all_successful_outputs_verified"]
    assert first_summary["all_pool_best_references_proven_optimal"]
    assert (first / "resolved_manifest.json").exists()
    assert (first / "records.csv").exists()
    assert (first / "aggregate.csv").exists()
    figures = export_run_figures(first)
    assert all(
        Path(path).exists()
        for paths in figures["figures"].values()
        for path in paths
    )
    records = [
        json.loads(line)
        for line in (first / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all("exact_evaluation_seconds" in record["metrics"] for record in records)
    assert all("selection_seconds" in record["metrics"] for record in records)

    aggregate = aggregate_run_directories(
        (first, second),
        output=tmp_path / "multi_seed.json",
    )
    assert aggregate["runs"] == 2
    assert aggregate["unique_seeds"] == 1
    assert aggregate["aggregation_scope"] == "repeated_runs"
    assert (tmp_path / "multi_seed.csv").exists()
    assert set(aggregate["methods"]) == {"fallback", "random"}
    assert "fallback__vs__random" in aggregate["pairwise"]
    assert "test_id" in aggregate["partitions"]
    assert "fallback__vs__random" in aggregate["pairwise_by_partition"]["test_id"]


def test_milp_method_requires_a_real_time_limit() -> None:
    with pytest.raises(ValueError, match="requires time_limit_seconds"):
        MethodSpec(method_id="milp", kind="milp_time_limit").validate()


def test_direct_method_requires_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="requires a checkpoint"):
        MethodSpec(method_id="direct", kind="direct_hybrid").validate()


def test_quality_fingerprint_ignores_timing_noise() -> None:
    base = {
        "instance_id": "i",
        "method_id": "m",
        "method_seed": 1,
        "success": True,
        "source": "raw",
        "objective": 1.0,
        "gap_to_pool_best": 0.0,
        "metrics": {
            "raw_feasible_count": 1,
            "repair_successes": 0,
            "fallback_success": False,
            "total_seconds": 1.0,
        },
    }
    changed = json.loads(json.dumps(base))
    changed["metrics"]["total_seconds"] = 99.0
    assert quality_fingerprint([base]) == quality_fingerprint([changed])


def test_pairwise_outcomes_preserve_failures_and_quality_wins() -> None:
    records = [
        {"instance_id": "a", "method_id": "left", "success": True, "objective": 1.0},
        {"instance_id": "a", "method_id": "right", "success": True, "objective": 2.0},
        {"instance_id": "b", "method_id": "left", "success": False, "objective": None},
        {"instance_id": "b", "method_id": "right", "success": True, "objective": 3.0},
    ]

    outcome = pairwise_outcomes(records)["left__vs__right"]

    assert outcome["left_wins"] == 1
    assert outcome["right_only_success"] == 1
    assert outcome["instances"] == 2


def test_time_matched_budget_selects_smallest_direct_k_at_threshold() -> None:
    def aggregate(direct: dict[int, float], diffusion: float = 1.0) -> dict:
        methods = {
            "diffusion_k4": {"total_seconds": {"mean": diffusion}},
        }
        methods.update(
            {
                f"direct_k{count}": {"total_seconds": {"mean": value}}
                for count, value in direct.items()
            }
        )
        return {"methods": methods}

    selection = select_time_matched_budget(
        {
            "initial": aggregate({4: 0.1, 64: 0.8}),
            "extension": aggregate({128: 0.55, 256: 1.0}, diffusion=0.5),
        },
        time_match_ratio=0.9,
    )

    assert selection["selected_direct_k"] == 128
    assert selection["threshold_reached"]
    assert selection["selection_rule"] == "smallest_k_reaching_threshold"


def test_phase6ee_stage1_gate_prefers_trajectory_rescue_when_signal_exists() -> None:
    snapshot = {
        "transition_index": 0,
        "timestep": 100,
        "previous_timestep": 96,
        "state_use": {
            "shuffle_js": 0.02,
            "shuffle_argmax_change": 0.08,
            "shuffle_total_variation": 0.1,
            "perturb_js": 0.01,
            "perturb_argmax_change": 0.06,
            "perturb_total_variation": 0.08,
        },
        "dependency_response": {
            "target_js_sum": 0.2,
            "target_count": 1,
            "neighbor_js_sum": 0.3,
            "neighbor_count": 2,
            "non_neighbor_js_sum": 0.1,
            "non_neighbor_count": 2,
            "competitor_js_sum": 0.1,
            "competitor_count": 2,
            "unrelated_js_sum": 0.1,
            "unrelated_count": 2,
        },
        "sampled_state": {"any_feasible": False, "best_gap_to_pool_best": None},
        "clean_argmax": {"any_feasible": True, "best_gap_to_pool_best": 0.05},
    }
    records = [
        {
            "seed": 1,
            "diagnostic_seconds": 1.0,
            "snapshots": [snapshot],
            "final": {"any_feasible": False, "best_gap_to_pool_best": None},
            "reservoir": {"any_feasible": True, "best_gap_to_pool_best": 0.05},
            "reservoir_improves_final": True,
        }
    ]
    aggregate = aggregate_phase6ee_stage1_records(
        records,
        thresholds={
            "state_argmax_change_min": 0.05,
            "neighbor_response_ratio_min": 1.2,
            "reservoir_raw_any_gain_min": 0.05,
            "reservoir_gap_reduction_min": 0.1,
        },
    )

    assert aggregate["gate_r1"]["localized_dependency_use"]
    assert aggregate["gate_r1"]["trajectory_signal"]
    assert aggregate["gate_r1"]["recommendation"] == "stage2_trajectory_rescue"


def test_phase6ee_stage2a_split_selection_and_gate_are_deterministic() -> None:
    from gdm_factor_diffusion.experiments import (
        evaluate_gate_r2,
        select_stage2a_variant,
        stable_validation_split,
    )

    ids = [f"validation-{index:05d}" for index in range(8)]
    first = stable_validation_split(ids, calibration_count=4)
    second = stable_validation_split(reversed(ids), calibration_count=4)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])

    aggregate = {
        "methods": {
            "diffusion_final_k4": {
                "final_success_rate": 1.0,
                "gap_to_pool_best": 0.020,
                "total_seconds": 1.0,
                "raw_any_feasible_rate": 0.55,
                "best_pre_fallback_gap_to_pool_best": 0.030,
            },
            "direct_k96": {
                "final_success_rate": 1.0,
                "gap_to_pool_best": 0.005,
                "total_seconds": 1.0,
                "raw_any_feasible_rate": 0.85,
                "best_pre_fallback_gap_to_pool_best": 0.010,
            },
            "rescue_three_anchor_b8": {
                "final_success_rate": 1.0,
                "gap_to_pool_best": 0.010,
                "total_seconds": 1.05,
                "raw_any_feasible_rate": 0.65,
                "best_pre_fallback_gap_to_pool_best": 0.020,
            },
            "rescue_all_five_b12": {
                "final_success_rate": 1.0,
                "gap_to_pool_best": 0.008,
                "total_seconds": 1.20,
                "raw_any_feasible_rate": 0.70,
                "best_pre_fallback_gap_to_pool_best": 0.018,
            },
        }
    }
    selection = select_stage2a_variant(aggregate, max_direct_time_ratio=1.10)
    assert selection["selected_variant"] == "rescue_three_anchor_b8"
    gate = evaluate_gate_r2(
        aggregate,
        selected_variant=selection["selected_variant"],
        thresholds={
            "minimum_gap_closure": 0.50,
            "max_direct_time_ratio": 1.10,
            "minimum_raw_any_gain": 0.05,
        },
    )
    assert gate["passed"]
    assert gate["recommendation"] == "sealed_final_id_rescue_authorized"


def test_phase6ee_stage2b_quality_replay_ignores_timing_only() -> None:
    from gdm_factor_diffusion.experiments import (
        quality_fingerprint as stage2b_quality_fingerprint,
        quality_payload as stage2b_quality_payload,
    )

    entry = {
        "success": True,
        "source": "repair",
        "objective": 2.0,
        "gap_to_pool_best": 0.1,
        "metrics": {
            "raw_any_feasible": False,
            "raw_feasible_count": 0,
            "repair_attempts": 4,
            "repair_successes": 2,
            "final_success": True,
            "total_seconds": 1.0,
        },
    }
    changed = json.loads(json.dumps(entry))
    changed["metrics"]["total_seconds"] = 99.0
    first = stage2b_quality_payload(entry, method_seed=7, pool_best=1.8)
    second = stage2b_quality_payload(changed, method_seed=7, pool_best=1.8)
    assert first == second
    references = {(1, "test_id-00000", "direct_k96"): first}
    assert stage2b_quality_fingerprint(
        references, method_id="direct_k96"
    ) == stage2b_quality_fingerprint(references, method_id="direct_k96")


def test_phase6ee_stage2b_interpretation_and_pairing_are_explicit() -> None:
    from gdm_factor_diffusion.experiments import (
        interpret_stage2b,
        phase6ee_stage2b_paired_outcomes,
    )

    records = [
        {
            "methods": {
                "diffusion_final_k4": {"objective": 2.0},
                "rescue_all_five_b12": {"objective": 1.5},
                "direct_k96": {"objective": 1.0},
            }
        },
        {
            "methods": {
                "diffusion_final_k4": {"objective": 1.0},
                "rescue_all_five_b12": {"objective": 1.0},
                "direct_k96": {"objective": 1.0},
            }
        },
    ]
    paired = phase6ee_stage2b_paired_outcomes(
        records, "diffusion_final_k4", "rescue_all_five_b12"
    )
    assert paired == {"left_wins": 0, "ties": 1, "right_wins": 1}

    def metrics(gap: float, seconds: float, raw: float, fallback: float) -> dict:
        return {
            "gap_to_pool_best": gap,
            "total_seconds": seconds,
            "raw_any_feasible_rate": raw,
            "selected_source_rates": {"fallback": fallback},
        }

    aggregate = {
        "methods": {
            "diffusion_final_k4": metrics(0.02, 1.0, 0.5, 0.6),
            "rescue_all_five_b12": metrics(0.01, 1.0, 0.7, 0.4),
            "direct_k96": metrics(0.005, 1.0, 0.9, 0.3),
        }
    }
    interpretation = interpret_stage2b(
        aggregate, selected_variant="rescue_all_five_b12"
    )
    assert interpretation["rescue_improves_final_k4"]
    assert not interpretation["rescue_beats_direct_k96"]
    assert interpretation["gap_closure"] == pytest.approx(2 / 3)
    assert (
        interpretation["conclusion"]
        == "trajectory_rescue_improves_diffusion_but_direct_remains_stronger"
    )


def test_scale_probe_config_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError, match="MILP time limit"):
        ScaleProbeConfig(milp_time_limit_seconds=0).validate()
    with pytest.raises(ValueError, match="training_profile_steps"):
        ScaleProbeConfig(training_profile_steps=0).validate()


def test_training_curve_export_uses_append_only_metrics(tmp_path: Path) -> None:
    records = [
        {"split": "train", "step": 1, "loss_total": 1.0},
        {
            "split": "validation_denoising",
            "step": 1,
            "loss_total": 1.1,
            "clean_accuracy": 0.5,
        },
        {
            "split": "validation_constrained",
            "step": 1,
            "mean_gap_to_pool_best": 0.1,
            "mean_raw_feasible_rate": 0.25,
            "learned_wins_over_fallback": 1,
            "instances": 2,
        },
    ]
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    payload = export_training_curves(tmp_path)

    assert payload["scope"] == "single_seed_diagnostic"
    assert all(
        Path(path).exists()
        for paths in payload["figures"].values()
        for path in paths
    )


def test_phase6c_acceptance_run_passes_training_audit() -> None:
    run = (
        IMPLEMENTATION_ROOT
        / "artifacts"
        / "runs"
        / "20260615T070248Z-phase6c-final-acceptance"
    )

    audited = audit_training_run(run, expected_steps=1500)

    assert audited["seed"] == 20260622
    assert audited["best_step"] == 1300
    assert audited["best_metrics"]["verified_rate"] == pytest.approx(1.0)


def test_phase6c_checkpoint_freeze_hashes_are_valid() -> None:
    freeze = (
        IMPLEMENTATION_ROOT
        / "artifacts"
        / "phase6c-five-seed"
        / "checkpoint_freeze.json"
    )

    payload = verify_checkpoint_freeze(freeze)

    assert len(payload["seeds"]) == 5
    assert payload["expected_steps"] == 1500


def test_phase6d_c_sealed_campaign_lock_is_valid() -> None:
    lock = IMPLEMENTATION_ROOT / "artifacts" / "phase6d-c-final" / "campaign_lock.json"

    payload = verify_sealed_campaign_lock(lock, implementation_root=IMPLEMENTATION_ROOT)

    assert len(payload["manifests"]) == 12
    assert set(payload["datasets"]["main"]["partitions"]).isdisjoint(
        {"train", "validation"}
    )


def test_phase6d_c_final_evidence_hashes_are_valid() -> None:
    evidence = (
        IMPLEMENTATION_ROOT
        / "artifacts"
        / "phase6d-c-final"
        / "final_evidence_freeze.json"
    )

    payload = verify_phase6d_c_evidence(evidence)

    assert len(payload["runs"]) == 12
    assert set(payload["aggregates"]) == {"main", "scale"}


def test_phase6e_a_locked_campaign_is_valid() -> None:
    lock = (
        IMPLEMENTATION_ROOT
        / "artifacts"
        / "phase6e-a-inference"
        / "campaign_lock.json"
    )

    payload = verify_phase6e_a_lock(lock, implementation_root=IMPLEMENTATION_ROOT)

    assert len(payload["manifests"]) == 20
    assert {entry["group"] for entry in payload["manifests"]} == {
        "postprocessing",
        "reverse_steps",
        "proposal_count",
        "repair_max_moves",
    }


def test_phase6eb_locked_campaign_is_valid() -> None:
    lock = (
        IMPLEMENTATION_ROOT
        / "artifacts"
        / "phase6e-b-training"
        / "campaign_lock.json"
    )

    payload = verify_phase6eb_lock(lock, implementation_root=IMPLEMENTATION_ROOT)

    assert payload["seeds"] == [20260622, 20260623, 20260624]
    assert set(payload["variants"]) == {
        "energy_full",
        "uniform_full",
        "energy_no_guidance",
        "energy_no_capacity",
        "energy_no_link",
        "best_full",
    }
    assert sum(entry["mode"] == "full" for entry in payload["training_configs"]) == 15
    assert sum(entry["mode"] == "smoke" for entry in payload["training_configs"]) == 5
    for entry in payload["training_configs"]:
        config = load_config(IMPLEMENTATION_ROOT / entry["path"])
        assert config["training"]["train_partitions"] == ["train"]
        assert config["training"]["validation_partitions"] == ["validation"]


def test_phase6eb_final_evidence_hashes_are_valid() -> None:
    evidence = (
        IMPLEMENTATION_ROOT
        / "artifacts"
        / "phase6e-b-evaluation"
        / "final_evidence_freeze.json"
    )

    payload = verify_phase6eb_final_evidence(
        evidence, implementation_root=IMPLEMENTATION_ROOT
    )

    assert len(payload["runs"]) == 3
