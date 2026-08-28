"""Validation-only Phase 6E-E Stage 1 trajectory diagnostic campaign."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.inference import (
    TrajectoryDiagnosticConfig,
    diagnose_reverse_trajectory,
)
from gdm_factor_diffusion.training import (
    LabeledDeploymentDataset,
    audit_dataset_freeze,
)

from .runtime import load_learned_solver
from .schema import file_sha256
from .training_aggregation import verify_checkpoint_freeze

PHASE6EE_STAGE1_SCOPE = "phase6e_e_stage1_trajectory_diagnostics"
PHASE6EE_STAGE1_EVIDENCE_SCOPE = "phase6e_e_stage1_evidence"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checkpoint_map(
    freeze: Mapping[str, Any],
    root: Path,
    seeds: Iterable[int],
) -> dict[int, dict[str, Any]]:
    expected = set(int(seed) for seed in seeds)
    mapped: dict[int, dict[str, Any]] = {}
    for run in freeze["runs"]:
        seed = int(run["seed"])
        if seed not in expected:
            continue
        checkpoint = Path(run["best_checkpoint"]).resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if str(payload.get("model_kind", "diffusion")) != "diffusion":
            raise ValueError("Phase 6E-E Stage 1 requires diffusion checkpoints.")
        mapped[seed] = {
            "seed": seed,
            "checkpoint": _relative(root, checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
        }
    if set(mapped) != expected:
        raise ValueError("Diagnostic seeds do not match the checkpoint freeze.")
    return mapped


def prepare_phase6ee_stage1(
    config_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Lock validation instances, checkpoints, metrics, and Gate R1 thresholds."""

    root = Path(implementation_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)["diagnostic"]
    if str(config["schema_version"]) != "1.0":
        raise ValueError("Unsupported Phase 6E-E Stage 1 schema.")
    partition = str(config["partition"])
    if partition != "validation" or partition == str(config["final_partition"]):
        raise ValueError("Phase 6E-E Stage 1 is restricted to validation data.")
    seeds = tuple(int(seed) for seed in config["seeds"])
    if len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("Diagnostic seeds must be unique and nonempty.")

    dataset_root = _resolve(root, config["dataset_root"])
    audit_dataset_freeze(dataset_root)
    dataset_freeze = dataset_root / str(config["dataset_freeze"])
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=(partition,),
        require_freeze=True,
    )
    limit = min(int(config["instance_limit"]), len(dataset))
    if limit < 1:
        raise ValueError("instance_limit must select at least one instance.")
    instance_ids = [dataset[index].instance.instance_id for index in range(limit)]

    checkpoint_freeze = _resolve(root, config["checkpoint_freeze"])
    freeze = verify_checkpoint_freeze(checkpoint_freeze)
    checkpoints = _checkpoint_map(freeze, root, seeds)
    settings = {
        "num_samples": int(config["num_samples"]),
        "reverse_steps": int(config["reverse_steps"]),
        "anchor_count": int(config["anchor_count"]),
    }
    thresholds = {
        "state_argmax_change_min": float(
            config["thresholds"]["state_argmax_change_min"]
        ),
        "neighbor_response_ratio_min": float(
            config["thresholds"]["neighbor_response_ratio_min"]
        ),
        "reservoir_raw_any_gain_min": float(
            config["thresholds"]["reservoir_raw_any_gain_min"]
        ),
        "reservoir_gap_reduction_min": float(
            config["thresholds"]["reservoir_gap_reduction_min"]
        ),
    }
    lock = {
        "schema_version": "1.0",
        "scope": PHASE6EE_STAGE1_SCOPE,
        "config_path": _relative(root, config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_root": _relative(root, dataset_root),
        "dataset_freeze": _relative(root, dataset_freeze),
        "dataset_freeze_sha256": file_sha256(dataset_freeze),
        "checkpoint_freeze": _relative(root, checkpoint_freeze),
        "checkpoint_freeze_sha256": file_sha256(checkpoint_freeze),
        "partition": partition,
        "final_partition": str(config["final_partition"]),
        "seeds": list(seeds),
        "checkpoints": [checkpoints[seed] for seed in seeds],
        "instance_ids": instance_ids,
        "settings": settings,
        "thresholds": thresholds,
        "device": str(config["device"]),
        "deterministic": bool(config["deterministic"]),
        "output_root": str(config["output_root"]),
    }
    lock_path = _resolve(root, config["lock_path"])
    write_json(lock_path, lock)
    return verify_phase6ee_stage1_lock(lock_path, implementation_root=root)


def verify_phase6ee_stage1_lock(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PHASE6EE_STAGE1_SCOPE:
        raise ValueError("Unsupported Phase 6E-E Stage 1 lock scope.")
    if lock["partition"] != "validation" or lock["final_partition"] == "validation":
        raise ValueError("Stage 1 lock escaped the validation-only boundary.")
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("dataset_freeze", "dataset_freeze_sha256"),
        ("checkpoint_freeze", "checkpoint_freeze_sha256"),
    ):
        path = _resolve(root, lock[path_key])
        if file_sha256(path) != lock[hash_key]:
            raise ValueError(f"Stage 1 lock hash mismatch: {path}")
    dataset_root = _resolve(root, lock["dataset_root"])
    audit_dataset_freeze(dataset_root)
    freeze = verify_checkpoint_freeze(_resolve(root, lock["checkpoint_freeze"]))
    frozen_seeds = set(int(seed) for seed in freeze["seeds"])
    if not set(int(seed) for seed in lock["seeds"]).issubset(frozen_seeds):
        raise ValueError("Stage 1 uses a seed outside the checkpoint freeze.")
    for entry in lock["checkpoints"]:
        checkpoint = _resolve(root, entry["checkpoint"])
        if file_sha256(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError(f"Stage 1 checkpoint hash mismatch: {checkpoint}")
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=(lock["partition"],),
        require_freeze=True,
    )
    available = {dataset[index].instance.instance_id for index in range(len(dataset))}
    if not set(lock["instance_ids"]).issubset(available):
        raise ValueError("Stage 1 lock contains an unknown validation instance.")
    return lock


def run_phase6ee_stage1(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
    seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Run or resume per-instance diagnostics for selected frozen seeds."""

    root = Path(implementation_root).resolve()
    lock = verify_phase6ee_stage1_lock(lock_path, implementation_root=root)
    requested = (
        tuple(int(seed) for seed in seeds)
        if seeds is not None
        else tuple(int(seed) for seed in lock["seeds"])
    )
    unknown = set(requested) - set(int(seed) for seed in lock["seeds"])
    if unknown:
        raise ValueError(f"Unknown Stage 1 seeds: {sorted(unknown)}")
    target_device = torch.device(lock["device"])
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the locked Stage 1 configuration.")
    dataset = LabeledDeploymentDataset(
        _resolve(root, lock["dataset_root"]),
        partitions=(lock["partition"],),
        require_freeze=True,
    )
    items = {dataset[index].instance.instance_id: dataset[index] for index in range(len(dataset))}
    checkpoints = {int(entry["seed"]): entry for entry in lock["checkpoints"]}
    output_root = _resolve(root, lock["output_root"])
    settings = TrajectoryDiagnosticConfig(**lock["settings"])
    run_index: list[dict[str, Any]] = []

    for seed in requested:
        seed_everything(seed, deterministic=bool(lock["deterministic"]))
        checkpoint_entry = checkpoints[seed]
        checkpoint = _resolve(root, checkpoint_entry["checkpoint"])
        solver = load_learned_solver(checkpoint, dataset, target_device)
        if solver.model_kind != "diffusion":
            raise ValueError("Stage 1 loaded a non-diffusion checkpoint.")
        seed_root = output_root / f"seed{seed}"
        record_root = seed_root / "records"
        record_root.mkdir(parents=True, exist_ok=True)
        seed_start = perf_counter()
        completed = 0
        for instance_id in lock["instance_ids"]:
            record_path = record_root / f"{instance_id}.json"
            if record_path.exists():
                existing = _read_json(record_path)
                if (
                    existing.get("diagnostic_schema_version") != "1.1"
                ):
                    pass
                elif (
                    existing.get("seed") != seed
                    or existing.get("checkpoint_sha256")
                    != checkpoint_entry["checkpoint_sha256"]
                ):
                    raise ValueError(f"Stale Stage 1 record: {record_path}")
                else:
                    completed += 1
                    continue
            item = items[instance_id]
            diagnostic_seed = derive_seed(seed, f"phase6ee-stage1:{instance_id}")
            generator = torch.Generator(device=target_device).manual_seed(
                diagnostic_seed
            )
            result = diagnose_reverse_trajectory(
                solver.model,
                item.instance,
                solver.schedule,
                solver.feature_schema,
                reference_objective=float(np.min(item.pool.latencies)),
                config=settings,
                device=target_device,
                generator=generator,
            )
            result.update(
                {
                    "seed": seed,
                    "diagnostic_seed": diagnostic_seed,
                    "partition": lock["partition"],
                    "checkpoint_sha256": checkpoint_entry["checkpoint_sha256"],
                }
            )
            write_json(record_path, result)
            completed += 1
        summary = {
            "schema_version": "1.0",
            "scope": PHASE6EE_STAGE1_SCOPE,
            "seed": seed,
            "records": completed,
            "expected_records": len(lock["instance_ids"]),
            "checkpoint": checkpoint_entry["checkpoint"],
            "checkpoint_sha256": checkpoint_entry["checkpoint_sha256"],
            "elapsed_seconds": perf_counter() - seed_start,
        }
        summary_path = write_json(seed_root / "summary.json", summary)
        run_index.append(
            {
                "seed": seed,
                "summary": _relative(root, summary_path),
                "summary_sha256": file_sha256(summary_path),
                "records": completed,
            }
        )
    index = {
        "schema_version": "1.0",
        "scope": PHASE6EE_STAGE1_SCOPE,
        "lock": _relative(root, Path(lock_path).resolve()),
        "lock_sha256": file_sha256(lock_path),
        "runs": run_index,
    }
    write_json(output_root / "run_index.json", index)
    return index


def _mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    return float(np.mean(materialized)) if materialized else None


def aggregate_phase6ee_stage1_records(
    records: Iterable[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    records = list(records)
    if not records:
        raise ValueError("At least one Stage 1 record is required.")
    snapshots = [snapshot for record in records for snapshot in record["snapshots"]]
    state_use = {
        key: float(np.mean([snapshot["state_use"][key] for snapshot in snapshots]))
        for key in (
            "shuffle_js",
            "shuffle_argmax_change",
            "shuffle_total_variation",
            "perturb_js",
            "perturb_argmax_change",
            "perturb_total_variation",
        )
    }
    group_means: dict[str, float | None] = {}
    for group in (
        "target",
        "neighbor",
        "non_neighbor",
        "competitor",
        "unrelated",
    ):
        total = sum(
            snapshot["dependency_response"][f"{group}_js_sum"]
            for snapshot in snapshots
        )
        count = sum(
            snapshot["dependency_response"][f"{group}_count"]
            for snapshot in snapshots
        )
        group_means[group] = None if count == 0 else float(total / count)
    neighbor = group_means["neighbor"]
    unrelated = group_means["non_neighbor"]
    neighbor_ratio = (
        None
        if neighbor is None or unrelated is None
        else float(neighbor / max(unrelated, 1e-12))
    )

    final_any = float(np.mean([record["final"]["any_feasible"] for record in records]))
    reservoir_any = float(
        np.mean([record["reservoir"]["any_feasible"] for record in records])
    )
    final_gap = _mean(
        record["final"]["best_gap_to_pool_best"]
        for record in records
        if record["final"]["best_gap_to_pool_best"] is not None
    )
    reservoir_gap = _mean(
        record["reservoir"]["best_gap_to_pool_best"]
        for record in records
        if record["reservoir"]["best_gap_to_pool_best"] is not None
    )
    raw_any_gain = reservoir_any - final_any
    gap_reduction = (
        None
        if final_gap is None or reservoir_gap is None
        else float((final_gap - reservoir_gap) / max(abs(final_gap), 1e-12))
    )
    state_argmax_change = max(
        state_use["shuffle_argmax_change"],
        state_use["perturb_argmax_change"],
    )
    weak_state_use = state_argmax_change < thresholds["state_argmax_change_min"]
    localized_dependency_use = bool(
        neighbor_ratio is not None
        and neighbor_ratio >= thresholds["neighbor_response_ratio_min"]
    )
    trajectory_signal = bool(
        raw_any_gain >= thresholds["reservoir_raw_any_gain_min"]
        or (
            gap_reduction is not None
            and gap_reduction >= thresholds["reservoir_gap_reduction_min"]
        )
    )
    if weak_state_use:
        recommendation = "stage3_structural_redesign"
        rationale = "The denoiser is weakly sensitive to the noisy placement state."
    elif trajectory_signal:
        recommendation = "stage2_trajectory_rescue"
        rationale = "Intermediate clean/state candidates contain recoverable signal."
    else:
        recommendation = "stage3_structural_redesign"
        rationale = "The state is used, but the current reverse trajectory adds no useful candidates."

    by_transition: dict[str, Any] = {}
    for transition_index in sorted(
        {int(snapshot["transition_index"]) for snapshot in snapshots}
    ):
        selected = [
            snapshot
            for snapshot in snapshots
            if int(snapshot["transition_index"]) == transition_index
        ]
        first = selected[0]
        by_transition[str(transition_index)] = {
            "timestep": int(first["timestep"]),
            "previous_timestep": int(first["previous_timestep"]),
            "sampled_any_feasible_rate": float(
                np.mean([entry["sampled_state"]["any_feasible"] for entry in selected])
            ),
            "clean_argmax_any_feasible_rate": float(
                np.mean([entry["clean_argmax"]["any_feasible"] for entry in selected])
            ),
            "sampled_best_gap": _mean(
                entry["sampled_state"]["best_gap_to_pool_best"]
                for entry in selected
                if entry["sampled_state"]["best_gap_to_pool_best"] is not None
            ),
            "clean_argmax_best_gap": _mean(
                entry["clean_argmax"]["best_gap_to_pool_best"]
                for entry in selected
                if entry["clean_argmax"]["best_gap_to_pool_best"] is not None
            ),
        }
    return {
        "records": len(records),
        "unique_seeds": len({int(record["seed"]) for record in records}),
        "state_use": state_use,
        "dependency_response": {
            "mean_js": group_means,
            "neighbor_to_non_neighbor_ratio": neighbor_ratio,
        },
        "trajectory": {
            "final_raw_any_feasible_rate": final_any,
            "reservoir_raw_any_feasible_rate": reservoir_any,
            "reservoir_raw_any_gain": raw_any_gain,
            "final_conditional_best_gap": final_gap,
            "reservoir_conditional_best_gap": reservoir_gap,
            "reservoir_relative_gap_reduction": gap_reduction,
            "reservoir_improves_final_rate": float(
                np.mean([record["reservoir_improves_final"] for record in records])
            ),
            "mean_diagnostic_seconds": float(
                np.mean([record["diagnostic_seconds"] for record in records])
            ),
            "by_transition": by_transition,
        },
        "gate_r1": {
            "thresholds": dict(thresholds),
            "state_argmax_change": state_argmax_change,
            "weak_state_use": weak_state_use,
            "localized_dependency_use": localized_dependency_use,
            "trajectory_signal": trajectory_signal,
            "recommendation": recommendation,
            "rationale": rationale,
        },
    }


def finalize_phase6ee_stage1(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Aggregate all locked records and issue the predeclared Gate R1 decision."""

    root = Path(implementation_root).resolve()
    lock = verify_phase6ee_stage1_lock(lock_path, implementation_root=root)
    output_root = _resolve(root, lock["output_root"])
    records: list[dict[str, Any]] = []
    record_hashes: dict[str, str] = {}
    for seed in lock["seeds"]:
        checkpoint_hash = next(
            entry["checkpoint_sha256"]
            for entry in lock["checkpoints"]
            if int(entry["seed"]) == int(seed)
        )
        for instance_id in lock["instance_ids"]:
            path = output_root / f"seed{seed}" / "records" / f"{instance_id}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing Stage 1 record: {path}")
            record = _read_json(path)
            if (
                record["instance_id"] != instance_id
                or int(record["seed"]) != int(seed)
                or record["partition"] != "validation"
                or record["checkpoint_sha256"] != checkpoint_hash
            ):
                raise ValueError(f"Invalid Stage 1 record contract: {path}")
            records.append(record)
            record_hashes[_relative(root, path)] = file_sha256(path)
    aggregate = aggregate_phase6ee_stage1_records(
        records,
        thresholds=lock["thresholds"],
    )
    evidence = {
        "schema_version": "1.0",
        "scope": PHASE6EE_STAGE1_EVIDENCE_SCOPE,
        "lock": _relative(root, Path(lock_path).resolve()),
        "lock_sha256": file_sha256(lock_path),
        "partition": "validation",
        "seeds": list(lock["seeds"]),
        "instance_count": len(lock["instance_ids"]),
        "record_hashes": record_hashes,
        "aggregate": aggregate,
    }
    evidence_path = write_json(output_root / "stage1_evidence.json", evidence)
    report = _stage1_report(evidence)
    report_path = output_root / "STAGE1_REPORT_ZH.md"
    report_path.write_text(report, encoding="utf-8")
    evidence["evidence_path"] = _relative(root, evidence_path)
    evidence["report_path"] = _relative(root, report_path)
    return evidence


def _stage1_report(evidence: Mapping[str, Any]) -> str:
    aggregate = evidence["aggregate"]
    state = aggregate["state_use"]
    dependency = aggregate["dependency_response"]
    trajectory = aggregate["trajectory"]
    gate = aggregate["gate_r1"]
    ratio = dependency["neighbor_to_non_neighbor_ratio"]
    gap_reduction = trajectory["reservoir_relative_gap_reduction"]
    return "\n".join(
        (
            "# Phase 6E-E Stage 1 诊断报告",
            "",
            f"- 范围：validation，{evidence['instance_count']} 个实例，{len(evidence['seeds'])} 个冻结 seed。",
            f"- shuffle argmax change：{state['shuffle_argmax_change']:.4f}。",
            f"- local perturbation argmax change：{state['perturb_argmax_change']:.4f}。",
            f"- neighbor/non-neighbor response ratio：{'N/A' if ratio is None else f'{ratio:.4f}'}。",
            f"- final raw-any feasibility：{trajectory['final_raw_any_feasible_rate']:.4f}。",
            f"- trajectory reservoir raw-any feasibility：{trajectory['reservoir_raw_any_feasible_rate']:.4f}。",
            f"- reservoir relative gap reduction：{'N/A' if gap_reduction is None else f'{gap_reduction:.4f}'}。",
            f"- Gate R1 recommendation：`{gate['recommendation']}`。",
            f"- 解释：{gate['rationale']}",
            "",
            "该报告仅用于方法诊断。trajectory reservoir 使用多个中间锚点，候选预算高于正式 K=4，不能直接作为论文性能比较。",
            "",
        )
    )
