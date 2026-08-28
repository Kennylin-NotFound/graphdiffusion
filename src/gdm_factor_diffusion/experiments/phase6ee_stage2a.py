"""Validation-locked Phase 6E-E Stage 2A trajectory-rescue campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    SolveResult,
    build_trajectory_candidate_set,
    sample_direct_proposals,
    sample_reverse_trajectory_proposals,
    solve_from_proposals,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze

from .runtime import load_learned_solver
from .schema import file_sha256
from .training_aggregation import verify_checkpoint_freeze

PHASE6EE_STAGE2A_SCOPE = "phase6e_e_stage2a_trajectory_rescue"
PHASE6EE_STAGE2A_CALIBRATION_SCOPE = "phase6e_e_stage2a_calibration_freeze"
PHASE6EE_STAGE2A_EVIDENCE_SCOPE = "phase6e_e_stage2a_confirmation_evidence"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_validation_split(
    instance_ids: Iterable[str],
    *,
    calibration_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split validation IDs deterministically without using instance outcomes."""

    unique = tuple(dict.fromkeys(str(value) for value in instance_ids))
    if not 0 < calibration_count < len(unique):
        raise ValueError("calibration_count must leave both split halves nonempty.")
    ordered = tuple(
        sorted(
            unique,
            key=lambda value: (
                hashlib.sha256(value.encode("utf-8")).hexdigest(),
                value,
            ),
        )
    )
    return ordered[:calibration_count], ordered[calibration_count:]


def _checkpoint_map(
    freeze: Mapping[str, Any],
    root: Path,
    seeds: Iterable[int],
    *,
    expected_kind: str,
) -> dict[int, dict[str, Any]]:
    expected = set(int(seed) for seed in seeds)
    mapped: dict[int, dict[str, Any]] = {}
    for run in freeze["runs"]:
        seed = int(run["seed"])
        if seed not in expected:
            continue
        checkpoint = Path(run["best_checkpoint"]).resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        kind = str(payload.get("model_kind", "diffusion"))
        if kind != expected_kind:
            raise ValueError(f"Expected {expected_kind} checkpoint: {checkpoint}")
        mapped[seed] = {
            "seed": seed,
            "checkpoint": _relative(root, checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
        }
    if set(mapped) != expected:
        raise ValueError(f"{expected_kind} checkpoint seeds do not match Stage 2A.")
    return mapped


def _variant_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    anchor_sets = {
        str(name): tuple(int(value) for value in values)
        for name, values in config["anchor_sets"].items()
    }
    if "final_only" not in anchor_sets or anchor_sets["final_only"]:
        raise ValueError("anchor_sets must define an empty final_only baseline.")
    reverse_steps = int(config["diffusion"]["reverse_steps"])
    if any(
        index < 0 or index >= reverse_steps
        for values in anchor_sets.values()
        for index in values
    ):
        raise ValueError("Anchor indices must lie within the reverse trajectory.")
    repair_limits = tuple(int(value) for value in config["repair_candidate_limits"])
    if not repair_limits or any(value < 1 for value in repair_limits):
        raise ValueError("repair_candidate_limits must be positive and nonempty.")
    variants = [
        {
            "method_id": "diffusion_final_k4",
            "anchor_set": "final_only",
            "anchor_indices": [],
            "repair_candidate_limit": None,
        }
    ]
    for name, anchors in anchor_sets.items():
        if name == "final_only":
            continue
        for limit in repair_limits:
            variants.append(
                {
                    "method_id": f"rescue_{name}_b{limit}",
                    "anchor_set": name,
                    "anchor_indices": list(anchors),
                    "repair_candidate_limit": limit,
                }
            )
    return variants


def prepare_phase6ee_stage2a(
    config_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Freeze Stage 2A validation splits, variants, checkpoints, and gates."""

    root = Path(implementation_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)["stage2a"]
    if str(config["schema_version"]) != "1.0":
        raise ValueError("Unsupported Phase 6E-E Stage 2A schema.")
    partition = str(config["partition"])
    if partition != "validation" or partition == str(config["final_partition"]):
        raise ValueError("Stage 2A is restricted to validation data.")
    seeds = tuple(int(seed) for seed in config["seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Stage 2A seeds must be unique and nonempty.")

    dataset_root = _resolve(root, config["dataset_root"])
    audit_dataset_freeze(dataset_root)
    dataset_freeze = dataset_root / str(config["dataset_freeze"])
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=(partition,),
        require_freeze=True,
    )
    limit = min(int(config["instance_limit"]), len(dataset))
    instance_ids = tuple(dataset[index].instance.instance_id for index in range(limit))
    calibration_ids, confirmation_ids = stable_validation_split(
        instance_ids,
        calibration_count=int(config["calibration_count"]),
    )

    diffusion_freeze_path = _resolve(root, config["diffusion_checkpoint_freeze"])
    direct_freeze_path = _resolve(root, config["direct_checkpoint_freeze"])
    diffusion_freeze = verify_checkpoint_freeze(diffusion_freeze_path)
    direct_freeze = verify_checkpoint_freeze(direct_freeze_path)
    diffusion = _checkpoint_map(
        diffusion_freeze, root, seeds, expected_kind="diffusion"
    )
    direct = _checkpoint_map(direct_freeze, root, seeds, expected_kind="direct")
    variants = _variant_specs(config)
    output_root = _resolve(root, config["output_root"])
    lock = {
        "schema_version": "1.0",
        "scope": PHASE6EE_STAGE2A_SCOPE,
        "config_path": _relative(root, config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_root": _relative(root, dataset_root),
        "dataset_freeze": _relative(root, dataset_freeze),
        "dataset_freeze_sha256": file_sha256(dataset_freeze),
        "diffusion_checkpoint_freeze": _relative(root, diffusion_freeze_path),
        "diffusion_checkpoint_freeze_sha256": file_sha256(diffusion_freeze_path),
        "direct_checkpoint_freeze": _relative(root, direct_freeze_path),
        "direct_checkpoint_freeze_sha256": file_sha256(direct_freeze_path),
        "partition": partition,
        "final_partition": str(config["final_partition"]),
        "seeds": list(seeds),
        "diffusion_checkpoints": [diffusion[seed] for seed in seeds],
        "direct_checkpoints": [direct[seed] for seed in seeds],
        "calibration_instance_ids": list(calibration_ids),
        "confirmation_instance_ids": list(confirmation_ids),
        "variants": variants,
        "diffusion": dict(config["diffusion"]),
        "direct": dict(config["direct"]),
        "postprocessing": dict(config["postprocessing"]),
        "gate_r2": dict(config["gate_r2"]),
        "device": str(config["device"]),
        "deterministic": bool(config["deterministic"]),
        "output_root": _relative(root, output_root),
    }
    lock_path = _resolve(root, config["lock_path"])
    write_json(lock_path, lock)
    return verify_phase6ee_stage2a_lock(lock_path, implementation_root=root)


def verify_phase6ee_stage2a_lock(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PHASE6EE_STAGE2A_SCOPE:
        raise ValueError("Unsupported Stage 2A lock scope.")
    if lock["partition"] != "validation" or lock["final_partition"] == "validation":
        raise ValueError("Stage 2A lock escaped the validation-only boundary.")
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("dataset_freeze", "dataset_freeze_sha256"),
        ("diffusion_checkpoint_freeze", "diffusion_checkpoint_freeze_sha256"),
        ("direct_checkpoint_freeze", "direct_checkpoint_freeze_sha256"),
    ):
        path = _resolve(root, lock[path_key])
        if file_sha256(path) != lock[hash_key]:
            raise ValueError(f"Stage 2A lock hash mismatch: {path}")
    audit_dataset_freeze(_resolve(root, lock["dataset_root"]))
    verify_checkpoint_freeze(_resolve(root, lock["diffusion_checkpoint_freeze"]))
    verify_checkpoint_freeze(_resolve(root, lock["direct_checkpoint_freeze"]))
    calibration = set(lock["calibration_instance_ids"])
    confirmation = set(lock["confirmation_instance_ids"])
    if calibration & confirmation or not calibration or not confirmation:
        raise ValueError("Stage 2A validation splits are not disjoint and nonempty.")
    for group in ("diffusion_checkpoints", "direct_checkpoints"):
        for entry in lock[group]:
            path = _resolve(root, entry["checkpoint"])
            if file_sha256(path) != entry["checkpoint_sha256"]:
                raise ValueError(f"Stage 2A checkpoint hash mismatch: {path}")
    return lock


def _inference_config(
    lock: Mapping[str, Any],
    *,
    num_samples: int,
    repair_candidate_limit: int | None,
    reverse_steps: int | None,
) -> InferenceConfig:
    post = lock["postprocessing"]
    settings = InferenceConfig(
        num_samples=num_samples,
        sample_batch_size=int(
            lock["diffusion"]["sample_batch_size"]
            if reverse_steps is not None
            else lock["direct"]["sample_batch_size"]
        ),
        repair_max_moves=int(post["repair_max_moves"]),
        fallback_max_search_nodes=int(post["fallback_max_search_nodes"]),
        enable_repair=True,
        enable_fallback=True,
        always_include_fallback=True,
        reverse_steps=reverse_steps,
        repair_candidate_limit=repair_candidate_limit,
    )
    settings.validate()
    return settings


def _result_payload(result: SolveResult, pool_best: float) -> dict[str, Any]:
    metrics = dict(result.metrics)
    for objective_key, gap_key in (
        ("best_raw_objective", "best_raw_gap_to_pool_best"),
        ("best_pre_fallback_objective", "best_pre_fallback_gap_to_pool_best"),
    ):
        value = metrics.get(objective_key)
        metrics[gap_key] = None if value is None else float(value) / pool_best - 1.0
    return {
        "success": result.success,
        "source": result.source,
        "objective": result.objective,
        "gap_to_pool_best": (
            None if result.objective is None else result.objective / pool_best - 1.0
        ),
        "metrics": metrics,
    }


def run_trajectory_diffusion_methods(
    lock: Mapping[str, Any],
    solver,
    item,
    *,
    selected_variant: str | None,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    variants = list(lock["variants"])
    if selected_variant is not None:
        variants = [
            variant
            for variant in variants
            if variant["method_id"] in {"diffusion_final_k4", selected_variant}
        ]
    anchor_union = tuple(
        sorted(
            {
                int(index)
                for variant in variants
                for index in variant["anchor_indices"]
            }
        )
    )
    diffusion = lock["diffusion"]
    base_config = _inference_config(
        lock,
        num_samples=int(diffusion["num_samples"]),
        repair_candidate_limit=None,
        reverse_steps=int(diffusion["reverse_steps"]),
    )
    trajectory = sample_reverse_trajectory_proposals(
        solver.model,
        item.instance,
        solver.schedule,
        solver.feature_schema,
        anchor_indices=anchor_union,
        config=base_config,
        device=device,
        generator=generator,
    )
    pool_best = float(np.min(item.pool.latencies))
    methods: dict[str, Any] = {}
    for variant in variants:
        method_id = str(variant["method_id"])
        anchors = tuple(int(index) for index in variant["anchor_indices"])
        if method_id == "diffusion_final_k4":
            proposals = trajectory.final_proposals
            probabilities = trajectory.final_probabilities
            preparation_seconds = 0.0
            before_dedup = int(proposals.shape[0])
            sources = ["final"] * proposals.shape[0]
        else:
            candidates = build_trajectory_candidate_set(
                trajectory,
                anchor_indices=anchors,
            )
            proposals = candidates.proposals
            probabilities = candidates.probabilities
            preparation_seconds = candidates.preparation_seconds
            before_dedup = candidates.candidates_before_deduplication
            sources = list(candidates.sources)
        config = _inference_config(
            lock,
            num_samples=int(proposals.shape[0]),
            repair_candidate_limit=variant["repair_candidate_limit"],
            reverse_steps=int(diffusion["reverse_steps"]),
        )
        result = solve_from_proposals(
            item.instance,
            proposals,
            model_probabilities=probabilities,
            config=config,
            sampling_seconds=trajectory.sampling_seconds,
            proposal_preparation_seconds=preparation_seconds,
            proposal_method=method_id,
        )
        result.metrics.update(
            {
                "trajectory_anchor_indices": list(anchors),
                "trajectory_candidate_count_before_deduplication": before_dedup,
                "trajectory_candidate_count_after_deduplication": int(
                    proposals.shape[0]
                ),
                "trajectory_candidate_sources": sources,
            }
        )
        methods[method_id] = _result_payload(result, pool_best)
    return methods


def run_direct_method(
    lock: Mapping[str, Any],
    solver,
    item,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    direct = lock["direct"]
    config = _inference_config(
        lock,
        num_samples=int(direct["num_samples"]),
        repair_candidate_limit=None,
        reverse_steps=None,
    )
    proposals, probabilities, sampling_seconds = sample_direct_proposals(
        solver.model,
        item.instance,
        solver.feature_schema,
        config=config,
        device=device,
        generator=generator,
    )
    result = solve_from_proposals(
        item.instance,
        proposals,
        model_probabilities=probabilities,
        config=config,
        sampling_seconds=sampling_seconds,
        proposal_method="direct_k96",
    )
    return _result_payload(result, float(np.min(item.pool.latencies)))


def _record_path(output_root: Path, split: str, seed: int, instance_id: str) -> Path:
    return output_root / split / f"seed{seed}" / "records" / f"{instance_id}.json"


def _verify_existing_record(
    record: Mapping[str, Any],
    *,
    split: str,
    seed: int,
    diffusion_hash: str,
    direct_hash: str,
    expected_methods: set[str],
) -> bool:
    return bool(
        record.get("schema_version") == "1.0"
        and record.get("split") == split
        and int(record.get("seed", -1)) == seed
        and record.get("partition") == "validation"
        and record.get("diffusion_checkpoint_sha256") == diffusion_hash
        and record.get("direct_checkpoint_sha256") == direct_hash
        and set(record.get("methods", {})) == expected_methods
    )


def run_phase6ee_stage2a(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
    split: str,
    seeds: Iterable[int] | None = None,
    calibration_freeze_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run or resume calibration or confirmation on its locked validation half."""

    if split not in {"calibration", "confirmation"}:
        raise ValueError("split must be calibration or confirmation.")
    root = Path(implementation_root).resolve()
    lock = verify_phase6ee_stage2a_lock(lock_path, implementation_root=root)
    selected_variant = None
    if split == "confirmation":
        if calibration_freeze_path is None:
            raise ValueError("confirmation requires a calibration freeze.")
        calibration = verify_phase6ee_stage2a_calibration(
            calibration_freeze_path,
            lock_path=lock_path,
            implementation_root=root,
        )
        selected_variant = str(calibration["selected_variant"])
    requested = (
        tuple(int(seed) for seed in seeds)
        if seeds is not None
        else tuple(int(seed) for seed in lock["seeds"])
    )
    if not set(requested) <= set(int(seed) for seed in lock["seeds"]):
        raise ValueError("Requested seed is outside the Stage 2A lock.")
    device = torch.device(lock["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the locked Stage 2A campaign.")
    dataset = LabeledDeploymentDataset(
        _resolve(root, lock["dataset_root"]),
        partitions=(lock["partition"],),
        require_freeze=True,
    )
    items = {dataset[index].instance.instance_id: dataset[index] for index in range(len(dataset))}
    diffusion_entries = {int(entry["seed"]): entry for entry in lock["diffusion_checkpoints"]}
    direct_entries = {int(entry["seed"]): entry for entry in lock["direct_checkpoints"]}
    instance_ids = lock[f"{split}_instance_ids"]
    expected_methods = (
        {"diffusion_final_k4", "direct_k96", selected_variant}
        if selected_variant is not None
        else {"direct_k96"} | {str(value["method_id"]) for value in lock["variants"]}
    )
    output_root = _resolve(root, lock["output_root"])
    completed = 0
    for seed in requested:
        seed_everything(seed, deterministic=bool(lock["deterministic"]))
        diffusion_entry = diffusion_entries[seed]
        direct_entry = direct_entries[seed]
        diffusion_solver = load_learned_solver(
            _resolve(root, diffusion_entry["checkpoint"]), dataset, device
        )
        direct_solver = load_learned_solver(
            _resolve(root, direct_entry["checkpoint"]), dataset, device
        )
        for instance_id in instance_ids:
            path = _record_path(output_root, split, seed, instance_id)
            if path.exists():
                existing = _read_json(path)
                if _verify_existing_record(
                    existing,
                    split=split,
                    seed=seed,
                    diffusion_hash=diffusion_entry["checkpoint_sha256"],
                    direct_hash=direct_entry["checkpoint_sha256"],
                    expected_methods=expected_methods,
                ):
                    completed += 1
                    continue
                raise ValueError(f"Stale Stage 2A record: {path}")
            item = items[instance_id]
            diffusion_seed = derive_seed(seed, f"phase6ee-stage2a:diffusion:{instance_id}")
            direct_seed = derive_seed(seed, f"phase6ee-stage2a:direct:{instance_id}")
            methods = run_trajectory_diffusion_methods(
                lock,
                diffusion_solver,
                item,
                selected_variant=selected_variant,
                generator=torch.Generator(device=device).manual_seed(diffusion_seed),
                device=device,
            )
            methods["direct_k96"] = run_direct_method(
                lock,
                direct_solver,
                item,
                generator=torch.Generator(device=device).manual_seed(direct_seed),
                device=device,
            )
            record = {
                "schema_version": "1.0",
                "scope": PHASE6EE_STAGE2A_SCOPE,
                "partition": "validation",
                "split": split,
                "seed": seed,
                "instance_id": instance_id,
                "diffusion_seed": diffusion_seed,
                "direct_seed": direct_seed,
                "diffusion_checkpoint_sha256": diffusion_entry["checkpoint_sha256"],
                "direct_checkpoint_sha256": direct_entry["checkpoint_sha256"],
                "pool_best": float(np.min(item.pool.latencies)),
                "methods": methods,
            }
            write_json(path, record)
            completed += 1
    return {
        "split": split,
        "seeds": list(requested),
        "instances_per_seed": len(instance_ids),
        "completed_records": completed,
        "selected_variant": selected_variant,
    }


def _mean(values: Iterable[float | int | bool | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return None if not selected else float(np.mean(selected))


def aggregate_phase6ee_stage2a_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    materialized = list(records)
    if not materialized:
        raise ValueError("Cannot aggregate empty Stage 2A records.")
    method_ids = set(materialized[0]["methods"])
    if any(set(record["methods"]) != method_ids for record in materialized):
        raise ValueError("Stage 2A records disagree on method IDs.")
    methods: dict[str, Any] = {}
    for method_id in sorted(method_ids):
        entries = [record["methods"][method_id] for record in materialized]
        metrics = [entry["metrics"] for entry in entries]
        source_counts = {
            source: sum(entry["source"] == source for entry in entries)
            for source in ("raw", "repair", "fallback", "failure")
        }
        methods[method_id] = {
            "records": len(entries),
            "final_success_rate": _mean(entry["success"] for entry in entries),
            "gap_to_pool_best": _mean(entry["gap_to_pool_best"] for entry in entries),
            "total_seconds": _mean(value["total_seconds"] for value in metrics),
            "raw_any_feasible_rate": _mean(value["raw_any_feasible"] for value in metrics),
            "raw_feasible_rate": _mean(value["raw_feasible_rate"] for value in metrics),
            "pre_fallback_success_rate": _mean(
                value["pre_fallback_success"] for value in metrics
            ),
            "best_pre_fallback_gap_to_pool_best": _mean(
                value["best_pre_fallback_gap_to_pool_best"] for value in metrics
            ),
            "fallback_invocation_rate": _mean(
                value["fallback_invoked"] for value in metrics
            ),
            "selected_source_counts": source_counts,
            "selected_source_rates": {
                source: count / len(entries) for source, count in source_counts.items()
            },
            "repair_attempts": _mean(value["repair_attempts"] for value in metrics),
            "raw_unique_count": _mean(value["raw_unique_count"] for value in metrics),
        }
    return {
        "records": len(materialized),
        "unique_seeds": len({int(record["seed"]) for record in materialized}),
        "methods": methods,
    }


def select_stage2a_variant(
    aggregate: Mapping[str, Any],
    *,
    max_direct_time_ratio: float,
) -> dict[str, Any]:
    """Select one rescue variant using calibration metrics only."""

    methods = aggregate["methods"]
    baseline = methods["diffusion_final_k4"]
    direct = methods["direct_k96"]
    candidates: list[dict[str, Any]] = []
    for method_id, metrics in methods.items():
        if not method_id.startswith("rescue_"):
            continue
        time_ratio = float(metrics["total_seconds"]) / float(direct["total_seconds"])
        eligible = bool(
            time_ratio <= max_direct_time_ratio
            and float(metrics["final_success_rate"])
            >= float(baseline["final_success_rate"]) - 1e-12
        )
        candidates.append(
            {
                "method_id": method_id,
                "eligible": eligible,
                "direct_time_ratio": time_ratio,
                **metrics,
            }
        )
    if not candidates:
        raise ValueError("Calibration aggregate contains no rescue variants.")
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    pool = eligible or candidates
    selected = min(
        pool,
        key=lambda value: (
            float("inf")
            if value["gap_to_pool_best"] is None
            else float(value["gap_to_pool_best"]),
            -float(value["raw_any_feasible_rate"]),
            float(value["total_seconds"]),
            value["method_id"],
        ),
    )
    return {
        "selected_variant": selected["method_id"],
        "selection_status": "eligible_min_gap" if eligible else "no_time_eligible_fallback",
        "max_direct_time_ratio": max_direct_time_ratio,
        "selected_metrics": selected,
        "eligible_variants": [value["method_id"] for value in eligible],
    }


def _collect_records(
    lock: Mapping[str, Any],
    root: Path,
    *,
    split: str,
    expected_methods: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    output_root = _resolve(root, lock["output_root"])
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for seed in lock["seeds"]:
        for instance_id in lock[f"{split}_instance_ids"]:
            path = _record_path(output_root, split, int(seed), instance_id)
            if not path.exists():
                raise FileNotFoundError(f"Missing Stage 2A record: {path}")
            record = _read_json(path)
            if (
                record.get("partition") != "validation"
                or record.get("split") != split
                or int(record.get("seed", -1)) != int(seed)
                or record.get("instance_id") != instance_id
            ):
                raise ValueError(f"Invalid Stage 2A record contract: {path}")
            if expected_methods is not None and set(record["methods"]) != expected_methods:
                raise ValueError(f"Unexpected confirmation methods: {path}")
            records.append(record)
            hashes[_relative(root, path)] = file_sha256(path)
    return records, hashes


def finalize_phase6ee_stage2a_calibration(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = verify_phase6ee_stage2a_lock(lock_path, implementation_root=root)
    records, hashes = _collect_records(lock, root, split="calibration")
    aggregate = aggregate_phase6ee_stage2a_records(records)
    selection = select_stage2a_variant(
        aggregate,
        max_direct_time_ratio=float(lock["gate_r2"]["max_direct_time_ratio"]),
    )
    payload = {
        "schema_version": "1.0",
        "scope": PHASE6EE_STAGE2A_CALIBRATION_SCOPE,
        "lock": _relative(root, Path(lock_path).resolve()),
        "lock_sha256": file_sha256(lock_path),
        "partition": "validation",
        "split": "calibration",
        "record_hashes": hashes,
        "aggregate": aggregate,
        **selection,
    }
    path = _resolve(root, lock["output_root"]) / "calibration_freeze.json"
    write_json(path, payload)
    return verify_phase6ee_stage2a_calibration(
        path,
        lock_path=lock_path,
        implementation_root=root,
    )


def verify_phase6ee_stage2a_calibration(
    calibration_path: str | Path,
    *,
    lock_path: str | Path,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = verify_phase6ee_stage2a_lock(lock_path, implementation_root=root)
    payload = _read_json(calibration_path)
    if payload.get("scope") != PHASE6EE_STAGE2A_CALIBRATION_SCOPE:
        raise ValueError("Unsupported Stage 2A calibration scope.")
    if payload["lock_sha256"] != file_sha256(lock_path):
        raise ValueError("Stage 2A calibration references a changed lock.")
    for relative, expected in payload["record_hashes"].items():
        if file_sha256(_resolve(root, relative)) != expected:
            raise ValueError(f"Stage 2A calibration record changed: {relative}")
    variants = {entry["method_id"] for entry in lock["variants"]}
    if payload["selected_variant"] not in variants - {"diffusion_final_k4"}:
        raise ValueError("Calibration selected an unknown or baseline variant.")
    return payload


def evaluate_gate_r2(
    aggregate: Mapping[str, Any],
    *,
    selected_variant: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    methods = aggregate["methods"]
    baseline = methods["diffusion_final_k4"]
    direct = methods["direct_k96"]
    rescue = methods[selected_variant]
    baseline_gap = float(baseline["gap_to_pool_best"])
    direct_gap = float(direct["gap_to_pool_best"])
    rescue_gap = float(rescue["gap_to_pool_best"])
    denominator = baseline_gap - direct_gap
    closure = None if denominator <= 0 else (baseline_gap - rescue_gap) / denominator
    time_ratio = float(rescue["total_seconds"]) / float(direct["total_seconds"])
    raw_gain = float(rescue["raw_any_feasible_rate"]) - float(
        baseline["raw_any_feasible_rate"]
    )
    baseline_pre_gap = baseline["best_pre_fallback_gap_to_pool_best"]
    rescue_pre_gap = rescue["best_pre_fallback_gap_to_pool_best"]
    pre_gap_improved = bool(
        rescue_pre_gap is not None
        and (baseline_pre_gap is None or float(rescue_pre_gap) < float(baseline_pre_gap))
    )
    checks = {
        "gap_closure": bool(
            closure is not None
            and closure >= float(thresholds["minimum_gap_closure"])
        ),
        "online_time": time_ratio <= float(thresholds["max_direct_time_ratio"]),
        "final_success": float(rescue["final_success_rate"])
        >= float(baseline["final_success_rate"]) - 1e-12,
        "pre_fallback": bool(
            raw_gain >= float(thresholds["minimum_raw_any_gain"])
            or pre_gap_improved
        ),
    }
    return {
        "selected_variant": selected_variant,
        "gap_closure": closure,
        "direct_time_ratio": time_ratio,
        "raw_any_gain": raw_gain,
        "pre_fallback_gap_improved": pre_gap_improved,
        "checks": checks,
        "passed": all(checks.values()),
        "recommendation": (
            "sealed_final_id_rescue_authorized"
            if all(checks.values())
            else "stage3_masked_partial_assignment_redesign"
        ),
    }


def finalize_phase6ee_stage2a_confirmation(
    lock_path: str | Path,
    calibration_freeze_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = verify_phase6ee_stage2a_lock(lock_path, implementation_root=root)
    calibration = verify_phase6ee_stage2a_calibration(
        calibration_freeze_path,
        lock_path=lock_path,
        implementation_root=root,
    )
    selected = str(calibration["selected_variant"])
    expected = {"diffusion_final_k4", "direct_k96", selected}
    records, hashes = _collect_records(
        lock,
        root,
        split="confirmation",
        expected_methods=expected,
    )
    aggregate = aggregate_phase6ee_stage2a_records(records)
    gate = evaluate_gate_r2(
        aggregate,
        selected_variant=selected,
        thresholds=lock["gate_r2"],
    )
    evidence = {
        "schema_version": "1.0",
        "scope": PHASE6EE_STAGE2A_EVIDENCE_SCOPE,
        "lock": _relative(root, Path(lock_path).resolve()),
        "lock_sha256": file_sha256(lock_path),
        "calibration_freeze": _relative(
            root, Path(calibration_freeze_path).resolve()
        ),
        "calibration_freeze_sha256": file_sha256(calibration_freeze_path),
        "partition": "validation",
        "split": "confirmation",
        "record_hashes": hashes,
        "aggregate": aggregate,
        "gate_r2": gate,
    }
    output_root = _resolve(root, lock["output_root"])
    evidence_path = output_root / "stage2a_evidence.json"
    write_json(evidence_path, evidence)
    report_path = output_root / "STAGE2A_REPORT_ZH.md"
    report_path.write_text(_stage2a_report(evidence), encoding="utf-8")
    return evidence


def _stage2a_report(evidence: Mapping[str, Any]) -> str:
    aggregate = evidence["aggregate"]
    gate = evidence["gate_r2"]
    selected = gate["selected_variant"]
    methods = aggregate["methods"]
    closure_text = (
        "N/A"
        if gate["gap_closure"] is None
        else f"{100 * gate['gap_closure']:.2f}%"
    )
    lines = [
        "# Phase 6E-E Stage 2A Confirmation 报告",
        "",
        "本报告仅使用 calibration 未见过的 validation confirmation 半区。",
        "",
        "| Method | Gap | Raw-any | Pre-fallback gap | Selected fallback | Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method_id in ("diffusion_final_k4", selected, "direct_k96"):
        value = methods[method_id]
        pre_gap = value["best_pre_fallback_gap_to_pool_best"]
        lines.append(
            f"| {method_id} | {100 * value['gap_to_pool_best']:.3f}% | "
            f"{100 * value['raw_any_feasible_rate']:.2f}% | "
            f"{'N/A' if pre_gap is None else f'{100 * pre_gap:.3f}%'} | "
            f"{100 * value['selected_source_rates']['fallback']:.2f}% | "
            f"{value['total_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gate R2",
            "",
            f"- Selected variant: `{selected}`",
            f"- Gap closure: {closure_text}",
            f"- Rescue/direct time ratio: {gate['direct_time_ratio']:.3f}",
            f"- Raw-any gain over final K=4: {100 * gate['raw_any_gain']:.2f} pp",
            f"- Passed: `{gate['passed']}`",
            f"- Recommendation: `{gate['recommendation']}`",
            "",
        ]
    )
    return "\n".join(lines)
