"""Shared evaluator for registered baselines and ablations."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from gdm_factor_diffusion.common.logging import (
    collect_run_metadata,
    create_run_directory,
    write_json,
)
from gdm_factor_diffusion.common.seed import seed_everything
from gdm_factor_diffusion.data.dataset import load_manifest
from gdm_factor_diffusion.training import LabeledDeploymentDataset
from gdm_factor_diffusion.solver import verify_placement

from .aggregation import (
    aggregate_records,
    quality_fingerprint,
    write_aggregate_csv,
    write_record_csv,
)
from .runtime import load_learned_solver, run_registered_method
from .schema import ExperimentManifest, file_sha256


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _pool_best_is_proven_optimal(item) -> bool:
    records = item.pool.metadata.get("solve_records", ())
    first = next(
        (record for record in records if record.get("rank_generated") == 0),
        None,
    )
    return bool(
        first is not None
        and first.get("optimal_under_exclusions")
        and abs(float(first["exact_objective"]) - float(item.pool.latencies[0])) <= 1e-8
    )


def _claim_report(manifest: ExperimentManifest, records: list[dict[str, Any]]) -> list[dict]:
    available_methods = {record["method_id"] for record in records}
    report = []
    for claim in manifest.claims:
        metric_covered = all(
            any(
                record["method_id"] == method
                and (
                    record.get(claim.primary_metric) is not None
                    or record["metrics"].get(claim.primary_metric) is not None
                )
                for record in records
            )
            for method in claim.comparison
        )
        report.append(
            {
                **asdict(claim),
                "methods_covered": set(claim.comparison) <= available_methods,
                "metric_covered": metric_covered,
                "status": (
                    "evidence_generated"
                    if set(claim.comparison) <= available_methods and metric_covered
                    else "incomplete"
                ),
            }
        )
    return report


def evaluate_experiment(
    manifest: ExperimentManifest,
    *,
    implementation_root: str | Path,
    output_root: str | Path | None = None,
) -> Path:
    manifest.validate()
    root = Path(implementation_root).resolve()
    dataset_root = _resolve(root, manifest.dataset_root).resolve()
    freeze_path = dataset_root / manifest.dataset_freeze
    if not freeze_path.exists():
        raise FileNotFoundError(f"Dataset freeze does not exist: {freeze_path}")

    seed_everything(manifest.seed, deterministic=manifest.deterministic)
    requested_device = torch.device(manifest.device)
    device = (
        requested_device
        if requested_device.type != "cuda" or torch.cuda.is_available()
        else torch.device("cpu")
    )
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=manifest.partitions,
        require_freeze=True,
    )
    dataset_manifest = load_manifest(dataset_root / "manifest.json")
    instance_metadata = {
        entry["instance_id"]: entry for entry in dataset_manifest["instances"]
    }
    count = len(dataset)
    if manifest.instance_limit is not None:
        count = min(count, manifest.instance_limit)

    destination_root = (
        Path(output_root)
        if output_root is not None
        else _resolve(root, manifest.output_root)
    )
    run_directory = create_run_directory(destination_root, manifest.name)

    checkpoint_paths = {
        method.method_id: (
            None
            if method.checkpoint is None
            else _resolve(root, method.checkpoint).resolve()
        )
        for method in manifest.methods
    }
    resolved = {
        "manifest": manifest.to_dict(),
        "manifest_fingerprint": manifest.fingerprint,
        "dataset_root": str(dataset_root),
        "dataset_freeze": str(freeze_path),
        "dataset_freeze_sha256": file_sha256(freeze_path),
        "dataset_core_sha256": json.loads(freeze_path.read_text(encoding="utf-8"))[
            "core_sha256"
        ],
        "device_requested": manifest.device,
        "device_resolved": str(device),
        "checkpoints": {
            method_id: (
                None
                if path is None
                else {"path": str(path), "sha256": file_sha256(path)}
            )
            for method_id, path in checkpoint_paths.items()
        },
        "instance_order": [
            dataset[index].instance.instance_id for index in range(count)
        ],
    }
    write_json(run_directory / "resolved_manifest.json", resolved)
    write_json(
        run_directory / "run_meta.json",
        collect_run_metadata(
            manifest.seed,
            resolved,
            project_root=root,
        ),
    )

    learned_cache = {}
    for method in manifest.methods:
        checkpoint_path = checkpoint_paths[method.method_id]
        if checkpoint_path is not None and checkpoint_path not in learned_cache:
            learned_cache[checkpoint_path] = load_learned_solver(
                checkpoint_path,
                dataset,
                device,
            )

    records: list[dict[str, Any]] = []
    for index in range(count):
        item = dataset[index]
        metadata = instance_metadata[item.instance.instance_id]
        pool_best = float(item.pool.latencies[0])
        for method in manifest.methods:
            checkpoint_path = checkpoint_paths[method.method_id]
            result, method_seed = run_registered_method(
                method,
                item,
                experiment_seed=manifest.seed,
                device=device,
                learned_solver=(
                    None
                    if checkpoint_path is None
                    else learned_cache[checkpoint_path]
                ),
            )
            objective = result.objective
            metrics = dict(result.metrics)
            for objective_metric, gap_metric in (
                ("best_raw_objective", "best_raw_gap_to_pool_best"),
                (
                    "best_pre_fallback_objective",
                    "best_pre_fallback_gap_to_pool_best",
                ),
            ):
                metric_value = metrics.get(objective_metric)
                metrics[gap_metric] = (
                    None
                    if metric_value is None
                    else float(metric_value) / pool_best - 1.0
                )
            output_verified = (
                None
                if result.placement is None
                else verify_placement(item.instance, result.placement).feasible
            )
            if method.time_limit_seconds is None:
                time_limit_scope = None
                time_limit_observed_seconds = None
                time_limit_exceeded = False
            elif method.kind == "milp_time_limit":
                time_limit_scope = "solver_optimization"
                time_limit_observed_seconds = float(
                    metrics["milp_solver_runtime_seconds"]
                )
                time_limit_exceeded = (
                    time_limit_observed_seconds > method.time_limit_seconds
                )
            else:
                time_limit_scope = "observed_total_online"
                time_limit_observed_seconds = float(metrics["total_seconds"])
                time_limit_exceeded = (
                    time_limit_observed_seconds > method.time_limit_seconds
                )
            records.append(
                {
                    "experiment_fingerprint": manifest.fingerprint,
                    "instance_id": item.instance.instance_id,
                    "partition": item.partition,
                    "regime": metadata["regime"],
                    "size_profile": metadata["size_profile"],
                    "num_services": item.instance.num_services,
                    "num_devices": item.instance.num_devices,
                    "num_dependencies": item.instance.num_dependencies,
                    "candidate_edges": int(item.instance.compatibility_mask.sum()),
                    "method_id": method.method_id,
                    "method_kind": method.kind,
                    "method_seed": method_seed,
                    "checkpoint_sha256": (
                        None
                        if checkpoint_path is None
                        else resolved["checkpoints"][method.method_id]["sha256"]
                    ),
                    "success": result.success,
                    "output_verified": output_verified,
                    "source": result.source,
                    "objective": objective,
                    "pool_best": pool_best,
                    "pool_best_proven_optimal": _pool_best_is_proven_optimal(item),
                    "gap_to_pool_best": (
                        None if objective is None else objective / pool_best - 1.0
                    ),
                    "time_limit_seconds": method.time_limit_seconds,
                    "time_limit_scope": time_limit_scope,
                    "time_limit_observed_seconds": time_limit_observed_seconds,
                    "time_limit_exceeded": time_limit_exceeded,
                    "metrics": metrics,
                }
            )

    records_path = run_directory / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    write_record_csv(run_directory / "records.csv", records)
    aggregate = aggregate_records(records)
    write_aggregate_csv(run_directory / "aggregate.csv", aggregate)
    summary = {
        "schema_version": "1.0",
        "run_directory": str(run_directory.resolve()),
        "manifest_fingerprint": manifest.fingerprint,
        "quality_fingerprint": quality_fingerprint(records),
        "records": len(records),
        "instances": count,
        "methods": len(manifest.methods),
        "all_final_outputs_verified": all(record["success"] for record in records),
        "all_successful_outputs_verified": all(
            record["output_verified"]
            for record in records
            if record["success"]
        ),
        "final_success_rate": sum(record["success"] for record in records) / len(records),
        "all_pool_best_references_proven_optimal": all(
            record["pool_best_proven_optimal"] for record in records
        ),
        "aggregate": aggregate,
        "claim_to_test": _claim_report(manifest, records),
    }
    write_json(run_directory / "summary.json", summary)
    return run_directory
