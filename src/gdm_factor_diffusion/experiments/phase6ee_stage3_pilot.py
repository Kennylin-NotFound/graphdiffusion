"""One-time Stage 3 pilot execution after checkpoint-only calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.inference import (
    MaskedDecodeConfig,
    solve_with_direct_predictor,
    solve_with_masked_model,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset

from .phase6ee_stage2a import run_trajectory_diffusion_methods
from .phase6ee_stage3 import (
    PILOT_CONTRACT_SCOPE,
    _aggregate_method,
    _historical_lock,
    _inference_config,
    _payload,
    _read_json,
    _relative,
    _resolve,
    load_stage3_solver,
    verify_stage3_preparation_lock,
)
from .runtime import load_learned_solver
from .schema import file_sha256

PILOT_EXECUTION_SCOPE = "phase6e_e_stage3_pilot_execution"
PILOT_RECORD_SCOPE = "phase6e_e_stage3_pilot_record"
PILOT_EVIDENCE_SCOPE = "phase6e_e_stage3_pilot_evidence"


def prepare_stage3_pilot_execution(
    contract_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    contract_path = Path(contract_path).resolve()
    contract = _read_json(contract_path)
    if contract.get("scope") != PILOT_CONTRACT_SCOPE:
        raise ValueError("Unsupported Stage 3 pilot contract.")
    if not contract.get("pilot_authorized"):
        raise ValueError("Checkpoint calibration did not authorize pilot access.")
    preparation_path = _resolve(root, contract["preparation_lock"])
    preparation = verify_stage3_preparation_lock(
        preparation_path, implementation_root=root
    )
    evidence_path = _resolve(root, contract["calibration_evidence"])
    if file_sha256(preparation_path) != contract["preparation_lock_sha256"]:
        raise ValueError("Pilot contract preparation hash mismatch.")
    if file_sha256(evidence_path) != contract["calibration_evidence_sha256"]:
        raise ValueError("Pilot contract calibration hash mismatch.")
    pilot_root = _resolve(root, contract["pilot_root"])
    if pilot_root.exists():
        raise ValueError("Pilot output already exists before execution lock.")
    source_paths = (
        "src/gdm_factor_diffusion/experiments/phase6ee_stage3_pilot.py",
        "scripts/69_run_phase6e_e_stage3_pilot.py",
    )
    execution = {
        "schema_version": "1.0",
        "scope": PILOT_EXECUTION_SCOPE,
        "contract": _relative(root, contract_path),
        "contract_sha256": file_sha256(contract_path),
        "preparation_lock": contract["preparation_lock"],
        "preparation_lock_sha256": contract["preparation_lock_sha256"],
        "calibration_evidence": contract["calibration_evidence"],
        "calibration_evidence_sha256": contract["calibration_evidence_sha256"],
        "source_sha256": {
            relative: file_sha256(root / relative) for relative in source_paths
        },
        "methods": list(contract["methods"]),
        "selected_direct_method": contract["selected_direct_method"],
        "gates": contract["gates"],
        "pilot_partition": contract["pilot_partition"],
        "pilot_instance_ids": list(contract["pilot_instance_ids"]),
        "pilot_root": contract["pilot_root"],
        "seed": int(preparation["seed"]),
        "device": preparation["device"],
    }
    destination = root / "artifacts" / "phase6e-e-stage3" / "pilot_execution_lock.json"
    write_json(destination, execution)
    return verify_stage3_pilot_execution_lock(destination, implementation_root=root)


def verify_stage3_pilot_execution_lock(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PILOT_EXECUTION_SCOPE:
        raise ValueError("Unsupported Stage 3 pilot execution lock.")
    for path_key, hash_key in (
        ("contract", "contract_sha256"),
        ("preparation_lock", "preparation_lock_sha256"),
        ("calibration_evidence", "calibration_evidence_sha256"),
    ):
        if file_sha256(_resolve(root, lock[path_key])) != lock[hash_key]:
            raise ValueError(f"Stage 3 pilot execution hash mismatch: {path_key}.")
    for relative, expected in lock["source_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Stage 3 pilot source hash mismatch: {relative}.")
    return lock


def _open_or_resume(lock: Mapping[str, Any], lock_path: Path, root: Path) -> Path:
    pilot_root = _resolve(root, lock["pilot_root"])
    marker = pilot_root / "opening_lock.json"
    expected = {
        "schema_version": "1.0",
        "scope": PILOT_EXECUTION_SCOPE,
        "execution_lock": _relative(root, lock_path),
        "execution_lock_sha256": file_sha256(lock_path),
        "pilot_instance_count": len(lock["pilot_instance_ids"]),
        "status": "opened_once_resumable",
    }
    if pilot_root.exists():
        if not marker.is_file() or _read_json(marker) != expected:
            raise ValueError("Pilot output exists without the matching opening lock.")
    else:
        pilot_root.mkdir(parents=True)
        write_json(marker, expected)
    return pilot_root


def _run_methods(
    item,
    preparation: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    direct_solver,
    masked_solver,
    historical_solver,
    device: torch.device,
) -> dict[str, Any]:
    pool_best = float(np.min(item.pool.latencies))
    direct_id = str(execution["selected_direct_method"])
    direct_count = int(direct_id.removeprefix("direct_k"))
    seed = int(execution["seed"])
    generator = torch.Generator(device=device).manual_seed(
        derive_seed(seed, f"{direct_id}:{item.instance.instance_id}")
    )
    direct = solve_with_direct_predictor(
        direct_solver.model,
        item.instance,
        direct_solver.feature_schema,
        config=_inference_config(preparation, direct_count),
        device=device,
        generator=generator,
    )
    methods = {direct_id: _payload(direct, pool_best)}

    stochastic_count = int(
        next(
            method_id.removeprefix("masked_stochastic_k")
            for method_id in execution["methods"]
            if method_id.startswith("masked_stochastic_k")
        )
    )
    for method_id, proposal_count, stochastic in (
        ("masked_deterministic_k1", 1, False),
        (f"masked_stochastic_k{stochastic_count}", stochastic_count, True),
    ):
        result = solve_with_masked_model(
            masked_solver.model,
            item.instance,
            masked_solver.schedule,
            masked_solver.feature_schema,
            decode_config=MaskedDecodeConfig(
                num_samples=proposal_count,
                sample_batch_size=min(
                    int(preparation["calibration"]["masked_sample_batch_size"]),
                    proposal_count,
                ),
                stochastic=stochastic,
                temperature=float(preparation["calibration"]["temperature"]),
            ),
            inference_config=_inference_config(preparation, proposal_count),
            device=device,
            generator=torch.Generator(device=device).manual_seed(
                derive_seed(seed, f"{method_id}:{item.instance.instance_id}")
            ),
        )
        methods[method_id] = _payload(result, pool_best)

    historical = run_trajectory_diffusion_methods(
        _historical_lock(preparation),
        historical_solver,
        item,
        selected_variant="rescue_all_five_b12",
        generator=torch.Generator(device=device).manual_seed(
            derive_seed(seed, f"historical:{item.instance.instance_id}")
        ),
        device=device,
    )
    for method_id, value in historical.items():
        metrics = value["metrics"]
        methods[method_id] = {
            "success": value["success"],
            "source": value["source"],
            "objective": value["objective"],
            "gap_to_pool_best": value["gap_to_pool_best"],
            "pre_fallback_success": bool(metrics["pre_fallback_success"]),
            "pre_fallback_gap": metrics["best_pre_fallback_gap_to_pool_best"],
            "raw_any_feasible": bool(metrics["raw_any_feasible"]),
            "raw_feasible_rate": metrics["raw_feasible_rate"],
            "fallback_invoked": bool(metrics["fallback_invoked"]),
            "sampling_seconds": float(metrics["sampling_seconds"]),
            "total_seconds": float(metrics["total_seconds"]),
            "selected_source": value["source"],
        }
    return methods


def run_stage3_pilot(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    execution = verify_stage3_pilot_execution_lock(
        lock_path, implementation_root=root
    )
    preparation = _read_json(_resolve(root, execution["preparation_lock"]))
    pilot_root = _open_or_resume(execution, lock_path, root)
    device = torch.device(execution["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Stage 3 pilot contract.")
    seed_everything(int(execution["seed"]), deterministic=True)
    dataset = LabeledDeploymentDataset(
        _resolve(root, preparation["dataset_root"]),
        partitions=(execution["pilot_partition"],),
        require_freeze=True,
    )
    items = {dataset[index].instance.instance_id: dataset[index] for index in range(len(dataset))}
    if set(items) != set(execution["pilot_instance_ids"]):
        raise ValueError("Loaded pilot IDs disagree with the execution lock.")
    direct_solver = load_stage3_solver(
        _resolve(root, preparation["checkpoints"]["direct"]["path"]), dataset, device
    )
    masked_solver = load_stage3_solver(
        _resolve(root, preparation["checkpoints"]["masked_conditional"]["path"]),
        dataset,
        device,
    )
    historical_solver = load_learned_solver(
        _resolve(root, preparation["checkpoints"]["historical_diffusion"]["path"]),
        dataset,
        device,
    )
    record_root = pilot_root / "records"
    expected_methods = set(execution["methods"])
    completed = 0
    for instance_id in execution["pilot_instance_ids"]:
        path = record_root / f"{instance_id}.json"
        if path.exists():
            existing = _read_json(path)
            if (
                existing.get("execution_lock_sha256") == file_sha256(lock_path)
                and set(existing.get("methods", {})) == expected_methods
            ):
                completed += 1
                continue
            raise ValueError(f"Stale Stage 3 pilot record: {path}")
        item = items[instance_id]
        methods = _run_methods(
            item,
            preparation,
            execution,
            direct_solver=direct_solver,
            masked_solver=masked_solver,
            historical_solver=historical_solver,
            device=device,
        )
        if set(methods) != expected_methods:
            raise ValueError("Stage 3 pilot method set drifted from the lock.")
        write_json(
            path,
            {
                "schema_version": "1.0",
                "scope": PILOT_RECORD_SCOPE,
                "partition": execution["pilot_partition"],
                "instance_id": instance_id,
                "pool_best": float(np.min(item.pool.latencies)),
                "execution_lock": _relative(root, lock_path),
                "execution_lock_sha256": file_sha256(lock_path),
                "methods": methods,
            },
        )
        completed += 1
    return {"instances": len(items), "completed": completed, "pilot_root": _relative(root, pilot_root)}


def _relative_improvement(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None or baseline <= 0:
        return None
    return (baseline - value) / baseline


def evaluate_stage3_gates(
    aggregate: Mapping[str, Mapping[str, Any]],
    records: list[Mapping[str, Any]],
    *,
    direct_id: str,
    stochastic_id: str,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    direct = aggregate[direct_id]
    deterministic = aggregate["masked_deterministic_k1"]
    stochastic = aggregate[stochastic_id]
    r3b_gap = _relative_improvement(
        direct["mean_pre_fallback_gap"], deterministic["mean_pre_fallback_gap"]
    )
    r3b_raw = 100.0 * (
        deterministic["raw_any_feasibility"] - direct["raw_any_feasibility"]
    )
    r3b_time = deterministic["mean_total_seconds"] / direct["mean_total_seconds"]
    r3b = {
        "relative_gap_improvement": r3b_gap,
        "raw_feasibility_percentage_point_improvement": r3b_raw,
        "time_ratio": r3b_time,
        "final_success_not_reduced": deterministic["final_success_rate"] >= direct["final_success_rate"],
    }
    r3b["passed"] = bool(
        r3b["final_success_not_reduced"]
        and r3b_time <= float(gates["partial_conditioning"]["maximum_time_ratio_to_direct"])
        and (
            (r3b_gap is not None and r3b_gap >= float(gates["partial_conditioning"]["minimum_relative_gap_improvement"]))
            or r3b_raw >= float(gates["partial_conditioning"]["minimum_raw_any_percentage_point_improvement"])
        )
    )

    wins = losses = ties = 0
    for record in records:
        det = record["methods"]["masked_deterministic_k1"]
        sto = record["methods"][stochastic_id]
        det_score = float("inf") if not det["pre_fallback_success"] else float(det["pre_fallback_gap"])
        sto_score = float("inf") if not sto["pre_fallback_success"] else float(sto["pre_fallback_gap"])
        if sto_score < det_score - 1e-12:
            wins += 1
        elif det_score < sto_score - 1e-12:
            losses += 1
        else:
            ties += 1
    r3c_gap = _relative_improvement(
        deterministic["mean_pre_fallback_gap"], stochastic["mean_pre_fallback_gap"]
    )
    r3c_time = stochastic["mean_total_seconds"] / deterministic["mean_total_seconds"]
    r3c = {
        "relative_gap_improvement": r3c_gap,
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "raw_feasibility_not_reduced": stochastic["raw_any_feasibility"] >= deterministic["raw_any_feasibility"],
        "time_ratio": r3c_time,
    }
    r3c["passed"] = bool(
        r3c_gap is not None
        and r3c_gap >= float(gates["diffusion_specific"]["minimum_relative_gap_improvement"])
        and wins > losses
        and r3c["raw_feasibility_not_reduced"]
        and r3c_time <= float(gates["diffusion_specific"]["maximum_time_ratio_to_deterministic"])
    )
    outcome = "A" if r3b["passed"] and r3c["passed"] else "B" if r3b["passed"] else "C"
    return {"gate_r3b": r3b, "gate_r3c": r3c, "outcome": outcome}


def finalize_stage3_pilot(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    execution = verify_stage3_pilot_execution_lock(lock_path, implementation_root=root)
    pilot_root = _resolve(root, execution["pilot_root"])
    records = [
        _read_json(pilot_root / "records" / f"{instance_id}.json")
        for instance_id in execution["pilot_instance_ids"]
    ]
    if any(record.get("scope") != PILOT_RECORD_SCOPE for record in records):
        raise ValueError("Invalid Stage 3 pilot record scope.")
    method_ids = list(execution["methods"])
    aggregate = {method_id: _aggregate_method(records, method_id) for method_id in method_ids}
    stochastic_id = next(
        method_id for method_id in method_ids if method_id.startswith("masked_stochastic_k")
    )
    gates = evaluate_stage3_gates(
        aggregate,
        records,
        direct_id=execution["selected_direct_method"],
        stochastic_id=stochastic_id,
        gates=execution["gates"],
    )
    evidence = {
        "schema_version": "1.0",
        "scope": PILOT_EVIDENCE_SCOPE,
        "execution_lock": _relative(root, lock_path),
        "execution_lock_sha256": file_sha256(lock_path),
        "partition": execution["pilot_partition"],
        "records": len(records),
        "aggregate": aggregate,
        **gates,
        "record_sha256": {
            record["instance_id"]: file_sha256(
                pilot_root / "records" / f"{record['instance_id']}.json"
            )
            for record in records
        },
    }
    destination = pilot_root / "pilot_evidence.json"
    if destination.exists() and _read_json(destination) != evidence:
        raise ValueError("Existing pilot evidence disagrees with current records.")
    write_json(destination, evidence)
    return {
        "evidence": _relative(root, destination),
        "evidence_sha256": file_sha256(destination),
        "outcome": gates["outcome"],
        "gate_r3b": gates["gate_r3b"],
        "gate_r3c": gates["gate_r3c"],
    }

