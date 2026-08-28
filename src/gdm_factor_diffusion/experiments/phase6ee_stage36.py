"""Checkpoint-only efficiency calibration after the consumed Stage 3 pilot."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.data.contracts import audit_dataset_config_contract
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    solve_with_masked_model,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze

from .phase6ee_stage3 import load_stage3_solver
from .schema import file_sha256

PREPARATION_SCOPE = "phase6e_e_stage36_efficiency_preparation"
RECORD_SCOPE = "phase6e_e_stage36_efficiency_record"
EVIDENCE_SCOPE = "phase6e_e_stage36_efficiency_evidence"
SELECTION_SCOPE = "phase6e_e_stage36_efficiency_selection"


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
        sample_batch_size=min(int(lock["calibration"]["sample_batch_size"]), count),
        repair_max_moves=int(post["repair_max_moves"]),
        fallback_max_search_nodes=int(post["fallback_max_search_nodes"]),
        enable_repair=True,
        enable_fallback=True,
        always_include_fallback=bool(post["always_include_fallback"]),
    )


def _result_payload(result, pool_best: float) -> dict[str, Any]:
    metrics = result.metrics
    pre = metrics.get("best_pre_fallback_objective")
    return {
        "success": result.success,
        "source": result.source,
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
    }


def _aggregate_method(
    records: Iterable[Mapping[str, Any]], method_id: str
) -> dict[str, Any]:
    entries = [record["methods"][method_id] for record in records]
    pre = [entry for entry in entries if entry["pre_fallback_success"]]
    return {
        "records": len(entries),
        "final_success_rate": mean(bool(entry["success"]) for entry in entries),
        "mean_gap_to_pool_best": mean(
            float(entry["gap_to_pool_best"])
            for entry in entries
            if entry["gap_to_pool_best"] is not None
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


def prepare_stage36_efficiency(
    config_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    evaluation = config["evaluation"]
    output = config["output"]
    destination = _resolve(root, output["preparation_lock"])
    if destination.exists():
        return verify_stage36_efficiency_lock(destination, implementation_root=root)

    dataset_root = _resolve(root, evaluation["dataset_root"])
    audit_dataset_freeze(dataset_root)
    manifest = _read_json(dataset_root / "manifest.json")
    partition = str(evaluation["calibration_partition"])
    instance_ids = [
        str(entry["instance_id"])
        for entry in manifest["instances"]
        if str(entry["partition"]) == partition
    ]
    if len(instance_ids) != 64:
        raise ValueError("Stage 3.6 calibration must use the locked 64-instance split.")

    pilot_path = _resolve(root, evaluation["consumed_pilot_evidence"])
    pilot = _read_json(pilot_path)
    if pilot.get("scope") != "phase6e_e_stage3_pilot_evidence":
        raise ValueError("Stage 3.6 requires the frozen consumed pilot evidence.")
    if pilot.get("outcome") != "B" or int(pilot.get("records", 0)) != 64:
        raise ValueError("Stage 3.6 expects the completed Outcome B pilot.")

    training_path = _resolve(root, evaluation["training_freeze"])
    training = _read_json(training_path)
    checkpoint_path = _resolve(root, evaluation["masked_checkpoint"])
    expected_checkpoint = training["runs"]["masked_conditional"]["sha256"]["best.pt"]
    if file_sha256(checkpoint_path).upper() != expected_checkpoint.upper():
        raise ValueError("Masked checkpoint differs from the Stage 3 training freeze.")

    forbidden_final = _resolve(root, evaluation["forbidden_final_root"])
    confirmation_root = _resolve(root, config["confirmation"]["dataset_root"])
    if forbidden_final.exists() or confirmation_root.exists():
        raise ValueError("Final or confirmation data was opened before selection freeze.")
    confirmation_config = _resolve(root, config["confirmation"]["dataset_config"])
    audit_dataset_config_contract(load_config(confirmation_config))

    source_paths = (
        "src/gdm_factor_diffusion/data/contracts.py",
        "src/gdm_factor_diffusion/experiments/phase6ee_stage36.py",
        "scripts/70_run_phase6e_e_stage36_efficiency.py",
        "configs/phase6e_e_stage36_efficiency.yaml",
        "configs/dataset_phase6e_e_stage36_confirmation.yaml",
    )
    lock = {
        "schema_version": "1.0",
        "scope": PREPARATION_SCOPE,
        "config": _relative(root, config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_root": _relative(root, dataset_root),
        "dataset_freeze": _relative(root, dataset_root / "dataset_freeze.json"),
        "dataset_freeze_sha256": file_sha256(dataset_root / "dataset_freeze.json"),
        "calibration_partition": partition,
        "calibration_instance_ids": instance_ids,
        "masked_checkpoint": _relative(root, checkpoint_path),
        "masked_checkpoint_sha256": file_sha256(checkpoint_path),
        "training_freeze": _relative(root, training_path),
        "training_freeze_sha256": file_sha256(training_path),
        "consumed_pilot_evidence": _relative(root, pilot_path),
        "consumed_pilot_evidence_sha256": file_sha256(pilot_path),
        "forbidden_final_root": str(evaluation["forbidden_final_root"]),
        "device": str(evaluation["device"]),
        "deterministic": bool(evaluation["deterministic"]),
        "seed": int(evaluation["seed"]),
        "calibration": config["calibration"],
        "postprocessing": config["postprocessing"],
        "selection_gate": config["selection_gate"],
        "confirmation": config["confirmation"],
        "calibration_root": str(output["calibration_root"]),
        "selection_lock": str(output["selection_lock"]),
        "source_sha256": {
            relative: file_sha256(root / relative) for relative in source_paths
        },
    }
    write_json(destination, lock)
    return verify_stage36_efficiency_lock(destination, implementation_root=root)


def verify_stage36_efficiency_lock(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PREPARATION_SCOPE:
        raise ValueError("Unsupported Stage 3.6 preparation lock.")
    for path_key, hash_key in (
        ("config", "config_sha256"),
        ("dataset_freeze", "dataset_freeze_sha256"),
        ("masked_checkpoint", "masked_checkpoint_sha256"),
        ("training_freeze", "training_freeze_sha256"),
        ("consumed_pilot_evidence", "consumed_pilot_evidence_sha256"),
    ):
        if file_sha256(_resolve(root, lock[path_key])) != lock[hash_key]:
            raise ValueError(f"Stage 3.6 hash mismatch: {path_key}.")
    for relative, expected in lock["source_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Stage 3.6 source hash mismatch: {relative}.")
    if _resolve(root, lock["forbidden_final_root"]).exists():
        raise ValueError("Stage 3 final data must remain closed.")
    if _resolve(root, lock["confirmation"]["dataset_root"]).exists():
        raise ValueError("Stage 3.6 confirmation data must remain closed during calibration.")
    return lock


def run_stage36_efficiency(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
    limit: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_stage36_efficiency_lock(lock_path, implementation_root=root)
    if smoke and (limit is None or limit > 2):
        raise ValueError("Smoke calibration is limited to at most two instances.")
    if not smoke and limit is not None:
        raise ValueError("Formal calibration must use all locked instances.")
    device = torch.device(lock["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Stage 3.6 contract.")
    seed_everything(int(lock["seed"]), deterministic=bool(lock["deterministic"]))
    dataset = LabeledDeploymentDataset(
        _resolve(root, lock["dataset_root"]),
        partitions=(lock["calibration_partition"],),
        require_freeze=True,
    )
    count = len(dataset) if limit is None else min(limit, len(dataset))
    solver = load_stage3_solver(
        _resolve(root, lock["masked_checkpoint"]), dataset, device
    )
    if solver.schedule is None or solver.model_kind != "masked_conditional":
        raise ValueError("Stage 3.6 requires the masked conditional checkpoint.")

    method_specs = [("masked_deterministic_k1", 1, False)] + [
        (f"masked_stochastic_k{int(k)}", int(k), True)
        for k in lock["calibration"]["stochastic_proposal_grid"]
    ]

    # Warm every batch shape outside the recorded timing scope.
    warm = dataset[0]
    for method_id, proposal_count, stochastic in method_specs:
        solve_with_masked_model(
            solver.model,
            warm.instance,
            solver.schedule,
            solver.feature_schema,
            decode_config=MaskedDecodeConfig(
                num_samples=proposal_count,
                sample_batch_size=min(
                    int(lock["calibration"]["sample_batch_size"]), proposal_count
                ),
                stochastic=stochastic,
                temperature=float(lock["calibration"]["temperature"]),
            ),
            inference_config=_inference_config(lock, proposal_count),
            device=device,
            generator=torch.Generator(device=device).manual_seed(
                derive_seed(int(lock["seed"]), f"warm:{method_id}")
            ),
        )

    output_root = _resolve(root, lock["calibration_root"])
    if smoke:
        output_root = output_root.parent / f"{output_root.name}-smoke"
    expected_methods = {spec[0] for spec in method_specs}
    completed = 0
    for index in range(count):
        item = dataset[index]
        path = output_root / "records" / f"{item.instance.instance_id}.json"
        if path.exists():
            existing = _read_json(path)
            if (
                existing.get("lock_sha256") == file_sha256(lock_path)
                and set(existing.get("methods", {})) == expected_methods
            ):
                completed += 1
                continue
            raise ValueError(f"Stale Stage 3.6 calibration record: {path}")
        pool_best = float(np.min(item.pool.latencies))
        methods: dict[str, Any] = {}
        shared_stochastic_seed = derive_seed(
            int(lock["seed"]), f"stochastic:{item.instance.instance_id}"
        )
        for method_id, proposal_count, stochastic in method_specs:
            generator_seed = (
                shared_stochastic_seed
                if stochastic
                else derive_seed(
                    int(lock["seed"]), f"deterministic:{item.instance.instance_id}"
                )
            )
            result = solve_with_masked_model(
                solver.model,
                item.instance,
                solver.schedule,
                solver.feature_schema,
                decode_config=MaskedDecodeConfig(
                    num_samples=proposal_count,
                    sample_batch_size=min(
                        int(lock["calibration"]["sample_batch_size"]),
                        proposal_count,
                    ),
                    stochastic=stochastic,
                    temperature=float(lock["calibration"]["temperature"]),
                ),
                inference_config=_inference_config(lock, proposal_count),
                device=device,
                generator=torch.Generator(device=device).manual_seed(generator_seed),
            )
            methods[method_id] = _result_payload(result, pool_best)
        write_json(
            path,
            {
                "schema_version": "1.0",
                "scope": RECORD_SCOPE,
                "partition": lock["calibration_partition"],
                "instance_id": item.instance.instance_id,
                "pool_best": pool_best,
                "lock": _relative(root, lock_path),
                "lock_sha256": file_sha256(lock_path),
                "methods": methods,
            },
        )
        completed += 1
    return {
        "smoke": smoke,
        "instances": count,
        "completed": completed,
        "output_root": _relative(root, output_root),
    }


def _relative_improvement(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None or baseline <= 0:
        return None
    return (baseline - value) / baseline


def evaluate_efficiency_candidates(
    aggregate: Mapping[str, Mapping[str, Any]],
    records: list[Mapping[str, Any]],
    *,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_id = "masked_deterministic_k1"
    baseline = aggregate[baseline_id]
    candidates: dict[str, Any] = {}
    for method_id in sorted(
        (name for name in aggregate if name.startswith("masked_stochastic_k")),
        key=lambda name: int(name.removeprefix("masked_stochastic_k")),
    ):
        current = aggregate[method_id]
        wins = losses = ties = 0
        for record in records:
            det = record["methods"][baseline_id]
            sto = record["methods"][method_id]
            det_score = (
                float("inf")
                if not det["pre_fallback_success"]
                else float(det["pre_fallback_gap"])
            )
            sto_score = (
                float("inf")
                if not sto["pre_fallback_success"]
                else float(sto["pre_fallback_gap"])
            )
            if sto_score < det_score - 1e-12:
                wins += 1
            elif det_score < sto_score - 1e-12:
                losses += 1
            else:
                ties += 1
        relative_gap = _relative_improvement(
            baseline["mean_pre_fallback_gap"], current["mean_pre_fallback_gap"]
        )
        total_ratio = current["mean_total_seconds"] / baseline["mean_total_seconds"]
        sampling_ratio = (
            current["mean_sampling_seconds"] / baseline["mean_sampling_seconds"]
        )
        raw_ok = current["raw_any_feasibility"] >= baseline["raw_any_feasibility"]
        success_ok = current["final_success_rate"] >= baseline["final_success_rate"]
        paired_ok = wins > losses
        passed = bool(
            relative_gap is not None
            and relative_gap >= float(gate["minimum_relative_gap_improvement"])
            and (paired_ok or not bool(gate["require_more_paired_wins_than_losses"]))
            and (raw_ok or not bool(gate["require_raw_feasibility_not_reduced"]))
            and (success_ok or not bool(gate["require_final_success_not_reduced"]))
            and total_ratio <= float(gate["maximum_time_ratio_to_deterministic"])
        )
        candidates[method_id] = {
            "proposal_count": int(method_id.removeprefix("masked_stochastic_k")),
            "relative_gap_improvement": relative_gap,
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
            "raw_feasibility_not_reduced": raw_ok,
            "final_success_not_reduced": success_ok,
            "total_time_ratio": total_ratio,
            "sampling_time_ratio": sampling_ratio,
            "passed": passed,
        }
    passing = [value for value in candidates.values() if value["passed"]]
    selected = (
        None
        if not passing
        else min(passing, key=lambda value: int(value["proposal_count"]))
    )
    return {
        "candidates": candidates,
        "selected_method": (
            None
            if selected is None
            else f"masked_stochastic_k{selected['proposal_count']}"
        ),
        "confirmation_authorized": selected is not None,
    }


def finalize_stage36_efficiency(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_stage36_efficiency_lock(lock_path, implementation_root=root)
    output_root = _resolve(root, lock["calibration_root"])
    records = [
        _read_json(output_root / "records" / f"{instance_id}.json")
        for instance_id in lock["calibration_instance_ids"]
    ]
    if any(record.get("scope") != RECORD_SCOPE for record in records):
        raise ValueError("Invalid Stage 3.6 record scope.")
    method_ids = sorted(records[0]["methods"])
    if any(set(record["methods"]) != set(method_ids) for record in records):
        raise ValueError("Stage 3.6 method sets disagree.")
    aggregate = {
        method_id: _aggregate_method(records, method_id) for method_id in method_ids
    }
    decision = evaluate_efficiency_candidates(
        aggregate, records, gate=lock["selection_gate"]
    )
    evidence = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "preparation_lock": _relative(root, lock_path),
        "preparation_lock_sha256": file_sha256(lock_path),
        "partition": lock["calibration_partition"],
        "records": len(records),
        "aggregate": aggregate,
        **decision,
        "record_sha256": {
            record["instance_id"]: file_sha256(
                output_root / "records" / f"{record['instance_id']}.json"
            )
            for record in records
        },
    }
    evidence_path = output_root / "efficiency_evidence.json"
    write_json(evidence_path, evidence)
    selection = {
        "schema_version": "1.0",
        "scope": SELECTION_SCOPE,
        "preparation_lock": _relative(root, lock_path),
        "preparation_lock_sha256": file_sha256(lock_path),
        "efficiency_evidence": _relative(root, evidence_path),
        "efficiency_evidence_sha256": file_sha256(evidence_path),
        "selected_method": decision["selected_method"],
        "confirmation_authorized": decision["confirmation_authorized"],
        "selection_policy": lock["selection_gate"]["policy"],
        "selection_gate": lock["selection_gate"],
        "confirmation": lock["confirmation"],
        "confirmation_data_generated": False,
        "consumed_pilot_evidence": lock["consumed_pilot_evidence"],
        "consumed_pilot_evidence_sha256": lock[
            "consumed_pilot_evidence_sha256"
        ],
    }
    selection_path = _resolve(root, lock["selection_lock"])
    write_json(selection_path, selection)
    return {
        "evidence": _relative(root, evidence_path),
        "evidence_sha256": file_sha256(evidence_path),
        "selection_lock": _relative(root, selection_path),
        "selection_lock_sha256": file_sha256(selection_path),
        **decision,
    }
