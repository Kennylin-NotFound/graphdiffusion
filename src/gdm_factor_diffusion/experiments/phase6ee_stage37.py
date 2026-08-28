"""Bounded decoder optimization and diffusion/direct matched-time gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    solve_with_direct_predictor,
    solve_with_masked_model,
)
from gdm_factor_diffusion.inference.masked_decode_vectorized import (
    solve_with_masked_model_vectorized,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze

from .phase6ee_stage3 import load_stage3_solver
from .schema import file_sha256

PREPARATION_SCOPE = "phase6e_e_stage37_preparation"
PREFLIGHT_SCOPE = "phase6e_e_stage37_optimization_preflight"
DECODER_SELECTION_SCOPE = "phase6e_e_stage37_decoder_selection"
RECORD_SCOPE = "phase6e_e_stage37_calibration_record"
EVIDENCE_SCOPE = "phase6e_e_stage37_calibration_evidence"
DECISION_SCOPE = "phase6e_e_stage37_matched_time_decision"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _inference_config(lock: Mapping[str, Any], count: int) -> InferenceConfig:
    post = lock["postprocessing"]
    return InferenceConfig(
        num_samples=count,
        sample_batch_size=min(int(lock["calibration"]["direct_sample_batch_size"]), count),
        repair_max_moves=int(post["repair_max_moves"]),
        fallback_max_search_nodes=int(post["fallback_max_search_nodes"]),
        enable_repair=True,
        enable_fallback=True,
        always_include_fallback=bool(post["always_include_fallback"]),
    )


def _payload(result, pool_best: float) -> dict[str, Any]:
    metrics = result.metrics
    pre = metrics.get("best_pre_fallback_objective")
    return {
        "success": result.success,
        "objective": result.objective,
        "gap_to_pool_best": (
            None if result.objective is None else float(result.objective) / pool_best - 1.0
        ),
        "pre_fallback_success": pre is not None,
        "pre_fallback_gap": None if pre is None else float(pre) / pool_best - 1.0,
        "raw_any_feasible": bool(metrics["raw_any_feasible"]),
        "raw_feasible_rate": float(metrics["raw_feasible_rate"]),
        "sampling_seconds": float(metrics["sampling_seconds"]),
        "total_seconds": float(metrics["total_seconds"]),
        "selected_source": result.source,
        "completed_rate": metrics.get("masked_completed_rate"),
    }


def _aggregate(records: Iterable[Mapping[str, Any]], method_id: str) -> dict[str, Any]:
    entries = [record["methods"][method_id] for record in records]
    pre = [entry for entry in entries if entry["pre_fallback_success"]]
    final = [entry for entry in entries if entry["success"]]
    return {
        "records": len(entries),
        "final_success_rate": len(final) / len(entries),
        "mean_gap_to_pool_best": (
            None
            if not final
            else mean(float(entry["gap_to_pool_best"]) for entry in final)
        ),
        "pre_fallback_success_rate": len(pre) / len(entries),
        "mean_pre_fallback_gap": (
            None
            if not pre
            else mean(float(entry["pre_fallback_gap"]) for entry in pre)
        ),
        "raw_any_feasibility": mean(
            bool(entry["raw_any_feasible"]) for entry in entries
        ),
        "mean_sampling_seconds": mean(
            float(entry["sampling_seconds"]) for entry in entries
        ),
        "mean_total_seconds": mean(float(entry["total_seconds"]) for entry in entries),
    }


def prepare_stage37(
    config_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    evaluation = config["evaluation"]
    output = config["output"]
    destination = _resolve(root, output["preparation_lock"])
    if destination.exists():
        return verify_stage37_lock(destination, implementation_root=root)

    dataset_root = _resolve(root, evaluation["dataset_root"])
    audit_dataset_freeze(dataset_root)
    manifest = _read_json(dataset_root / "manifest.json")
    partition = str(evaluation["partition"])
    ids = [
        str(entry["instance_id"])
        for entry in manifest["instances"]
        if str(entry["partition"]) == partition
    ]
    if len(ids) != 64:
        raise ValueError("Stage 3.7 requires all 64 checkpoint-selection IDs.")
    if _resolve(root, evaluation["forbidden_final_root"]).exists():
        raise ValueError("Stage 3 final data must remain closed.")

    training_path = _resolve(root, evaluation["training_freeze"])
    training = _read_json(training_path)
    direct_path = _resolve(root, evaluation["direct_checkpoint"])
    masked_path = _resolve(root, evaluation["masked_checkpoint"])
    expected = {
        "direct": training["runs"]["direct"]["sha256"]["best.pt"],
        "masked": training["runs"]["masked_conditional"]["sha256"]["best.pt"],
    }
    if file_sha256(direct_path).upper() != expected["direct"].upper():
        raise ValueError("Direct checkpoint differs from the training freeze.")
    if file_sha256(masked_path).upper() != expected["masked"].upper():
        raise ValueError("Masked checkpoint differs from the training freeze.")

    pilot_path = _resolve(root, evaluation["consumed_pilot_evidence"])
    stage36_path = _resolve(root, evaluation["stage36_evidence"])
    if _read_json(pilot_path).get("outcome") != "B":
        raise ValueError("Stage 3.7 requires the frozen Outcome B pilot.")
    if _read_json(stage36_path).get("confirmation_authorized"):
        raise ValueError("Stage 3.7 expects the closed Stage 3.6 efficiency branch.")

    source_paths = (
        "PHASE6E_E_STAGE37_PLAN.md",
        "src/gdm_factor_diffusion/inference/masked_decode_vectorized.py",
        "src/gdm_factor_diffusion/experiments/phase6ee_stage37.py",
        "scripts/71_run_phase6e_e_stage37_matched_time.py",
        "configs/phase6e_e_stage37_matched_time.yaml",
    )
    lock = {
        "schema_version": "1.0",
        "scope": PREPARATION_SCOPE,
        "config": _relative(root, config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_root": _relative(root, dataset_root),
        "dataset_freeze": _relative(root, dataset_root / "dataset_freeze.json"),
        "dataset_freeze_sha256": file_sha256(dataset_root / "dataset_freeze.json"),
        "partition": partition,
        "instance_ids": ids,
        "direct_checkpoint": _relative(root, direct_path),
        "direct_checkpoint_sha256": file_sha256(direct_path),
        "masked_checkpoint": _relative(root, masked_path),
        "masked_checkpoint_sha256": file_sha256(masked_path),
        "training_freeze": _relative(root, training_path),
        "training_freeze_sha256": file_sha256(training_path),
        "consumed_pilot_evidence": _relative(root, pilot_path),
        "consumed_pilot_evidence_sha256": file_sha256(pilot_path),
        "stage36_evidence": _relative(root, stage36_path),
        "stage36_evidence_sha256": file_sha256(stage36_path),
        "forbidden_final_root": str(evaluation["forbidden_final_root"]),
        "seed": int(evaluation["seed"]),
        "device": str(evaluation["device"]),
        "deterministic": bool(evaluation["deterministic"]),
        "optimization_preflight": config["optimization_preflight"],
        "calibration": config["calibration"],
        "postprocessing": config["postprocessing"],
        "matched_time_gate": config["matched_time_gate"],
        "output": config["output"],
        "source_sha256": {
            relative: file_sha256(root / relative) for relative in source_paths
        },
    }
    write_json(destination, lock)
    return verify_stage37_lock(destination, implementation_root=root)


def verify_stage37_lock(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PREPARATION_SCOPE:
        raise ValueError("Unsupported Stage 3.7 preparation lock.")
    for path_key, hash_key in (
        ("config", "config_sha256"),
        ("dataset_freeze", "dataset_freeze_sha256"),
        ("direct_checkpoint", "direct_checkpoint_sha256"),
        ("masked_checkpoint", "masked_checkpoint_sha256"),
        ("training_freeze", "training_freeze_sha256"),
        ("consumed_pilot_evidence", "consumed_pilot_evidence_sha256"),
        ("stage36_evidence", "stage36_evidence_sha256"),
    ):
        if file_sha256(_resolve(root, lock[path_key])) != lock[hash_key]:
            raise ValueError(f"Stage 3.7 hash mismatch: {path_key}.")
    for relative, expected in lock["source_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Stage 3.7 source hash mismatch: {relative}.")
    if _resolve(root, lock["forbidden_final_root"]).exists():
        raise ValueError("Stage 3 final data must remain closed.")
    return lock


def _masked_config(lock: Mapping[str, Any], count: int, stochastic: bool) -> MaskedDecodeConfig:
    return MaskedDecodeConfig(
        num_samples=count,
        sample_batch_size=min(int(lock["calibration"]["masked_sample_batch_size"]), count),
        stochastic=stochastic,
        temperature=float(lock["calibration"]["temperature"]),
    )


def _run_masked(
    solver,
    item,
    lock: Mapping[str, Any],
    *,
    count: int,
    stochastic: bool,
    seed: int,
    vectorized: bool,
    device: torch.device,
):
    solve: Callable[..., Any] = (
        solve_with_masked_model_vectorized if vectorized else solve_with_masked_model
    )
    return solve(
        solver.model,
        item.instance,
        solver.schedule,
        solver.feature_schema,
        decode_config=_masked_config(lock, count, stochastic),
        inference_config=_inference_config(lock, count),
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def run_stage37_preflight(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_stage37_lock(lock_path, implementation_root=root)
    device = torch.device(lock["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Stage 3.7 contract.")
    seed_everything(int(lock["seed"]), deterministic=bool(lock["deterministic"]))
    dataset = LabeledDeploymentDataset(
        _resolve(root, lock["dataset_root"]),
        partitions=(lock["partition"],),
        require_freeze=True,
    )
    solver = load_stage3_solver(_resolve(root, lock["masked_checkpoint"]), dataset, device)
    if solver.schedule is None:
        raise ValueError("Stage 3.7 requires the masked checkpoint.")
    count = int(lock["optimization_preflight"]["instance_count"])
    diffusion_samples = int(lock["optimization_preflight"]["diffusion_samples"])

    # Warm both decoder implementations before measuring them.
    warm = dataset[0]
    for vectorized in (False, True):
        _run_masked(
            solver,
            warm,
            lock,
            count=diffusion_samples,
            stochastic=True,
            seed=derive_seed(int(lock["seed"]), f"warm:{vectorized}"),
            vectorized=vectorized,
            device=device,
        )

    records = []
    for index in range(count):
        item = dataset[index]
        deterministic_seed = derive_seed(
            int(lock["seed"]), f"deterministic:{item.instance.instance_id}"
        )
        legacy_det = _run_masked(
            solver, item, lock, count=1, stochastic=False,
            seed=deterministic_seed, vectorized=False, device=device
        )
        vector_det = _run_masked(
            solver, item, lock, count=1, stochastic=False,
            seed=deterministic_seed, vectorized=True, device=device
        )
        exact_replay = bool(
            legacy_det.success == vector_det.success
            and legacy_det.objective == vector_det.objective
            and (
                legacy_det.placement is None
                or np.array_equal(legacy_det.placement, vector_det.placement)
            )
        )
        stochastic_seed = derive_seed(
            int(lock["seed"]), f"preflight:{item.instance.instance_id}"
        )
        legacy = _run_masked(
            solver, item, lock, count=diffusion_samples, stochastic=True,
            seed=stochastic_seed, vectorized=False, device=device
        )
        vectorized = _run_masked(
            solver, item, lock, count=diffusion_samples, stochastic=True,
            seed=stochastic_seed, vectorized=True, device=device
        )
        records.append(
            {
                "instance_id": item.instance.instance_id,
                "deterministic_exact_replay": exact_replay,
                "legacy_sampling_seconds": float(legacy.metrics["sampling_seconds"]),
                "legacy_total_seconds": float(legacy.metrics["total_seconds"]),
                "legacy_success": legacy.success,
                "vectorized_sampling_seconds": float(vectorized.metrics["sampling_seconds"]),
                "vectorized_total_seconds": float(vectorized.metrics["total_seconds"]),
                "vectorized_success": vectorized.success,
                "vectorized_completed_rate": float(
                    vectorized.metrics["masked_completed_rate"]
                ),
            }
        )
    legacy_sampling = mean(row["legacy_sampling_seconds"] for row in records)
    vector_sampling = mean(row["vectorized_sampling_seconds"] for row in records)
    ratio = vector_sampling / legacy_sampling
    preflight = lock["optimization_preflight"]
    accepted = bool(
        (
            all(row["deterministic_exact_replay"] for row in records)
            or not bool(preflight["require_exact_deterministic_replay"])
        )
        and all(row["vectorized_success"] for row in records)
        and (
            all(row["vectorized_completed_rate"] == 1.0 for row in records)
            or not bool(preflight["require_complete_vectorized_proposals"])
        )
        and ratio <= 1.0 - float(preflight["minimum_sampling_speedup"])
    )
    evidence = {
        "schema_version": "1.0",
        "scope": PREFLIGHT_SCOPE,
        "preparation_lock": _relative(root, lock_path),
        "preparation_lock_sha256": file_sha256(lock_path),
        "records": records,
        "mean_legacy_sampling_seconds": legacy_sampling,
        "mean_vectorized_sampling_seconds": vector_sampling,
        "vectorized_to_legacy_sampling_ratio": ratio,
        "vectorized_accepted": accepted,
    }
    evidence_path = _resolve(root, lock["output"]["optimization_preflight"])
    write_json(evidence_path, evidence)
    selection = {
        "schema_version": "1.0",
        "scope": DECODER_SELECTION_SCOPE,
        "preparation_lock": _relative(root, lock_path),
        "preparation_lock_sha256": file_sha256(lock_path),
        "optimization_preflight": _relative(root, evidence_path),
        "optimization_preflight_sha256": file_sha256(evidence_path),
        "selected_decoder": "vectorized" if accepted else "legacy",
    }
    selection_path = _resolve(root, lock["output"]["decoder_selection_lock"])
    write_json(selection_path, selection)
    return {**selection, "sampling_ratio": ratio}


def _verify_decoder_selection(lock: Mapping[str, Any], root: Path) -> dict[str, Any]:
    path = _resolve(root, lock["output"]["decoder_selection_lock"])
    selection = _read_json(path)
    if selection.get("scope") != DECODER_SELECTION_SCOPE:
        raise ValueError("Stage 3.7 decoder selection is missing or invalid.")
    evidence_path = _resolve(root, selection["optimization_preflight"])
    if file_sha256(evidence_path) != selection["optimization_preflight_sha256"]:
        raise ValueError("Stage 3.7 preflight evidence hash mismatch.")
    return selection


def run_stage37_calibration(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
    limit: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_stage37_lock(lock_path, implementation_root=root)
    selection = _verify_decoder_selection(lock, root)
    if smoke and (limit is None or limit > 2):
        raise ValueError("Smoke calibration is limited to two instances.")
    if not smoke and limit is not None:
        raise ValueError("Formal Stage 3.7 calibration must use all 64 instances.")
    vectorized = selection["selected_decoder"] == "vectorized"
    device = torch.device(lock["device"])
    seed_everything(int(lock["seed"]), deterministic=bool(lock["deterministic"]))
    dataset = LabeledDeploymentDataset(
        _resolve(root, lock["dataset_root"]),
        partitions=(lock["partition"],),
        require_freeze=True,
    )
    count = len(dataset) if limit is None else min(limit, len(dataset))
    direct = load_stage3_solver(_resolve(root, lock["direct_checkpoint"]), dataset, device)
    masked = load_stage3_solver(_resolve(root, lock["masked_checkpoint"]), dataset, device)
    if masked.schedule is None:
        raise ValueError("Stage 3.7 masked schedule is unavailable.")
    method_ids = {
        *(f"direct_k{int(k)}" for k in lock["calibration"]["direct_proposal_grid"]),
        "masked_deterministic_k1",
        "masked_diffusion_k8",
    }

    warm = dataset[0]
    for proposal_count in lock["calibration"]["direct_proposal_grid"]:
        solve_with_direct_predictor(
            direct.model, warm.instance, direct.feature_schema,
            config=_inference_config(lock, int(proposal_count)), device=device,
            generator=torch.Generator(device=device).manual_seed(
                derive_seed(int(lock["seed"]), f"warm:direct:{proposal_count}")
            ),
        )
    for samples, stochastic in ((1, False), (8, True)):
        _run_masked(
            masked, warm, lock, count=samples, stochastic=stochastic,
            seed=derive_seed(int(lock["seed"]), f"warm:masked:{samples}"),
            vectorized=vectorized, device=device,
        )

    output_root = _resolve(root, lock["output"]["calibration_root"])
    if smoke:
        output_root = output_root.parent / f"{output_root.name}-smoke"
    completed = 0
    for index in range(count):
        item = dataset[index]
        record_path = output_root / "records" / f"{item.instance.instance_id}.json"
        if record_path.exists():
            record = _read_json(record_path)
            if (
                record.get("preparation_lock_sha256") == file_sha256(lock_path)
                and record.get("selected_decoder") == selection["selected_decoder"]
                and set(record.get("methods", {})) == method_ids
            ):
                completed += 1
                continue
            raise ValueError(f"Stale Stage 3.7 record: {record_path}")
        pool_best = float(np.min(item.pool.latencies))
        methods: dict[str, Any] = {}
        for proposal_count in lock["calibration"]["direct_proposal_grid"]:
            proposal_count = int(proposal_count)
            method_id = f"direct_k{proposal_count}"
            result = solve_with_direct_predictor(
                direct.model, item.instance, direct.feature_schema,
                config=_inference_config(lock, proposal_count), device=device,
                generator=torch.Generator(device=device).manual_seed(
                    derive_seed(int(lock["seed"]), f"{method_id}:{item.instance.instance_id}")
                ),
            )
            methods[method_id] = _payload(result, pool_best)
        for method_id, samples, stochastic in (
            ("masked_deterministic_k1", 1, False),
            ("masked_diffusion_k8", 8, True),
        ):
            result = _run_masked(
                masked, item, lock, count=samples, stochastic=stochastic,
                seed=derive_seed(int(lock["seed"]), f"{method_id}:{item.instance.instance_id}"),
                vectorized=vectorized, device=device,
            )
            methods[method_id] = _payload(result, pool_best)
        write_json(
            record_path,
            {
                "schema_version": "1.0",
                "scope": RECORD_SCOPE,
                "partition": lock["partition"],
                "instance_id": item.instance.instance_id,
                "pool_best": pool_best,
                "preparation_lock": _relative(root, lock_path),
                "preparation_lock_sha256": file_sha256(lock_path),
                "selected_decoder": selection["selected_decoder"],
                "methods": methods,
            },
        )
        completed += 1
    return {
        "smoke": smoke,
        "instances": count,
        "completed": completed,
        "selected_decoder": selection["selected_decoder"],
        "output_root": _relative(root, output_root),
    }


def evaluate_matched_time_gate(
    aggregate: Mapping[str, Mapping[str, Any]],
    records: list[Mapping[str, Any]],
    *,
    selected_direct: str,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    direct = aggregate[selected_direct]
    diffusion = aggregate["masked_diffusion_k8"]
    time_ratio = diffusion["mean_total_seconds"] / direct["mean_total_seconds"]
    direct_pre = direct["mean_pre_fallback_gap"]
    diffusion_pre = diffusion["mean_pre_fallback_gap"]
    relative_pre = (
        None
        if direct_pre is None or diffusion_pre is None or direct_pre <= 0
        else (direct_pre - diffusion_pre) / direct_pre
    )
    wins = losses = ties = 0
    for record in records:
        left = record["methods"]["masked_diffusion_k8"]
        right = record["methods"][selected_direct]
        left_score = float("inf") if not left["pre_fallback_success"] else float(left["pre_fallback_gap"])
        right_score = float("inf") if not right["pre_fallback_success"] else float(right["pre_fallback_gap"])
        if left_score < right_score - 1e-12:
            wins += 1
        elif right_score < left_score - 1e-12:
            losses += 1
        else:
            ties += 1
    checks = {
        "time_matched": float(gate["minimum_time_ratio"]) <= time_ratio <= float(gate["maximum_time_ratio"]),
        "pre_fallback_gap_improved": relative_pre is not None and relative_pre >= float(gate["minimum_relative_pre_fallback_gap_improvement"]),
        "paired_wins_exceed_losses": wins > losses,
        "raw_feasibility_not_reduced": diffusion["raw_any_feasibility"] >= direct["raw_any_feasibility"],
        "final_gap_not_worse": diffusion["mean_gap_to_pool_best"] <= direct["mean_gap_to_pool_best"],
        "final_success_not_reduced": diffusion["final_success_rate"] >= direct["final_success_rate"],
    }
    passed = bool(
        checks["time_matched"]
        and checks["pre_fallback_gap_improved"]
        and (checks["paired_wins_exceed_losses"] or not bool(gate["require_more_paired_wins_than_losses"]))
        and (checks["raw_feasibility_not_reduced"] or not bool(gate["require_raw_feasibility_not_reduced"]))
        and (checks["final_gap_not_worse"] or not bool(gate["require_final_gap_not_worse"]))
        and (checks["final_success_not_reduced"] or not bool(gate["require_final_success_not_reduced"]))
    )
    return {
        "selected_direct_method": selected_direct,
        "time_ratio": time_ratio,
        "relative_pre_fallback_gap_improvement": relative_pre,
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "checks": checks,
        "passed": passed,
    }


def finalize_stage37(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_stage37_lock(lock_path, implementation_root=root)
    selection = _verify_decoder_selection(lock, root)
    output_root = _resolve(root, lock["output"]["calibration_root"])
    records = [
        _read_json(output_root / "records" / f"{instance_id}.json")
        for instance_id in lock["instance_ids"]
    ]
    if any(record.get("scope") != RECORD_SCOPE for record in records):
        raise ValueError("Invalid Stage 3.7 record scope.")
    method_ids = sorted(records[0]["methods"])
    aggregate = {method_id: _aggregate(records, method_id) for method_id in method_ids}
    diffusion_time = aggregate["masked_diffusion_k8"]["mean_total_seconds"]
    direct_ids = [name for name in method_ids if name.startswith("direct_k")]
    selected_direct = min(
        direct_ids,
        key=lambda name: abs(math.log(aggregate[name]["mean_total_seconds"] / diffusion_time)),
    )
    gate = evaluate_matched_time_gate(
        aggregate,
        records,
        selected_direct=selected_direct,
        gate=lock["matched_time_gate"],
    )
    evidence = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "preparation_lock": _relative(root, lock_path),
        "preparation_lock_sha256": file_sha256(lock_path),
        "selected_decoder": selection["selected_decoder"],
        "partition": lock["partition"],
        "records": len(records),
        "aggregate": aggregate,
        "matched_time_gate": gate,
        "sealed_multiseed_authorized": gate["passed"],
        "record_sha256": {
            record["instance_id"]: file_sha256(
                output_root / "records" / f"{record['instance_id']}.json"
            )
            for record in records
        },
    }
    evidence_path = output_root / "matched_time_evidence.json"
    write_json(evidence_path, evidence)
    decision = {
        "schema_version": "1.0",
        "scope": DECISION_SCOPE,
        "preparation_lock": _relative(root, lock_path),
        "preparation_lock_sha256": file_sha256(lock_path),
        "matched_time_evidence": _relative(root, evidence_path),
        "matched_time_evidence_sha256": file_sha256(evidence_path),
        "selected_decoder": selection["selected_decoder"],
        "selected_direct_method": selected_direct,
        "diffusion_method": "masked_diffusion_k8",
        "sealed_multiseed_authorized": gate["passed"],
        "paper_claim_if_confirmed": (
            "graph diffusion improves deployment quality over direct prediction "
            "under a comparable total online budget"
        ),
    }
    decision_path = _resolve(root, lock["output"]["decision_lock"])
    write_json(decision_path, decision)
    return {
        "evidence": _relative(root, evidence_path),
        "evidence_sha256": file_sha256(evidence_path),
        "decision_lock": _relative(root, decision_path),
        "decision_lock_sha256": file_sha256(decision_path),
        "selected_decoder": selection["selected_decoder"],
        **gate,
        "sealed_multiseed_authorized": gate["passed"],
    }

