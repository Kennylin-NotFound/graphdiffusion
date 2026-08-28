"""Batch solution-pool generation and auditing for immutable datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.data.dataset import load_manifest, load_manifest_instance
from gdm_factor_diffusion.data.schema import DeploymentInstance

from .latency_evaluator import evaluate_latency
from .placement_verifier import verify_placement
from .solution_pool import (
    SOLUTION_POOL_SCHEMA_VERSION,
    SolutionPool,
    SolutionPoolConfig,
    build_solution_pool,
    compute_energy_distribution,
    load_solution_pool,
    save_solution_pool,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_solution_pool(
    instance: DeploymentInstance,
    pool: SolutionPool,
    *,
    tolerance: float = 1e-8,
) -> None:
    """Cross-check every pool member against the shared verifier and evaluator."""

    if pool.instance_id != instance.instance_id:
        raise ValueError("Solution-pool instance_id does not match the instance.")
    if pool.placements.shape[1] != instance.num_services:
        raise ValueError("Solution-pool placements have the wrong service dimension.")
    for index, (placement, stored_latency) in enumerate(
        zip(pool.placements, pool.latencies, strict=True)
    ):
        verification = verify_placement(instance, placement)
        if not verification.feasible:
            raise ValueError(f"Pool member {index} failed verification.")
        exact_latency = evaluate_latency(instance, placement).objective
        if not np.isclose(exact_latency, stored_latency, atol=tolerance, rtol=tolerance):
            raise ValueError(
                f"Pool member {index} latency mismatch: "
                f"stored={stored_latency}, exact={exact_latency}."
            )
    expected_energy, expected_probability = compute_energy_distribution(
        pool.latencies,
        beta=float(pool.metadata["beta"]),
        epsilon=float(pool.metadata["energy_epsilon"]),
    )
    if not np.allclose(pool.normalized_energy, expected_energy, atol=tolerance):
        raise ValueError("Solution-pool normalized energies are inconsistent.")
    if not np.allclose(
        pool.sampling_probability,
        expected_probability,
        atol=tolerance,
    ):
        raise ValueError("Solution-pool sampling probabilities are inconsistent.")


def _load_existing_manifest(
    root: Path,
    dataset_manifest: Mapping[str, Any],
    config: SolutionPoolConfig,
) -> dict[str, Any] | None:
    path = root / "solution_pool_manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != SOLUTION_POOL_SCHEMA_VERSION:
        raise ValueError("Unsupported existing solution-pool manifest schema version.")
    if manifest.get("dataset_name") != dataset_manifest["dataset_name"]:
        raise ValueError("Existing solution-pool manifest refers to a different dataset.")
    if manifest.get("dataset_instance_count") != dataset_manifest["instance_count"]:
        raise ValueError("Existing solution-pool manifest has a stale dataset count.")
    if manifest.get("config") != asdict(config):
        raise ValueError(
            "Existing solution-pool manifest uses a different labeling configuration."
        )
    return manifest


def _pool_entry(
    root: Path,
    instance_entry: Mapping[str, Any],
    instance: DeploymentInstance,
    pool: SolutionPool,
    saved: Path,
    relative_path: Path,
) -> dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "partition": instance_entry["partition"],
        "instance_path": instance_entry["path"],
        "instance_sha256": instance_entry["sha256"],
        "pool_path": relative_path.as_posix(),
        "pool_sha256": _sha256(saved),
        "pool_size": pool.size,
        "minimum_latency": float(pool.latencies.min()),
        "maximum_latency": float(pool.latencies.max()),
        "termination_reason": pool.metadata["termination_reason"],
        "elapsed_seconds": pool.metadata["elapsed_seconds"],
    }


def _write_merged_manifest(
    root: Path,
    dataset_manifest: Mapping[str, Any],
    config: SolutionPoolConfig,
    entries_by_instance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ordered_entries = [
        dict(entries_by_instance[entry["instance_id"]])
        for entry in dataset_manifest["instances"]
        if entry["instance_id"] in entries_by_instance
    ]
    manifest = {
        "schema_version": SOLUTION_POOL_SCHEMA_VERSION,
        "dataset_name": dataset_manifest["dataset_name"],
        "dataset_manifest": "manifest.json",
        "dataset_instance_count": dataset_manifest["instance_count"],
        "labeled_instance_count": len(ordered_entries),
        "config": asdict(config),
        "pools": ordered_entries,
    }
    write_json(root / "solution_pool_manifest.json", manifest)
    return manifest


def _audit_existing_entry(
    root: Path,
    instance_entry: Mapping[str, Any],
    pool_entry: Mapping[str, Any],
) -> None:
    instance_id = instance_entry["instance_id"]
    if pool_entry.get("instance_sha256") != instance_entry["sha256"]:
        raise ValueError(f"Stale instance checksum for pool: {instance_id}")
    if pool_entry.get("instance_path") != instance_entry["path"]:
        raise ValueError(f"Stale instance path for pool: {instance_id}")
    if pool_entry.get("partition") != instance_entry["partition"]:
        raise ValueError(f"Stale partition for pool: {instance_id}")
    pool_path = root / pool_entry["pool_path"]
    if not pool_path.exists():
        raise ValueError(f"Missing existing solution pool: {pool_path}")
    if _sha256(pool_path) != pool_entry["pool_sha256"]:
        raise ValueError(f"Solution-pool checksum mismatch: {pool_path}")
    instance = load_manifest_instance(root, instance_entry)
    pool = load_solution_pool(pool_path)
    audit_solution_pool(instance, pool)


def _index_existing_entries(
    root: Path,
    dataset_manifest: Mapping[str, Any],
    existing_manifest: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if existing_manifest is None:
        return {}
    dataset_entries = {
        entry["instance_id"]: entry for entry in dataset_manifest["instances"]
    }
    indexed: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for raw_entry in existing_manifest.get("pools", []):
        entry = dict(raw_entry)
        instance_id = entry["instance_id"]
        if instance_id in indexed:
            raise ValueError("Existing solution-pool manifest contains duplicate IDs.")
        instance_entry = dataset_entries.get(instance_id)
        if instance_entry is None:
            raise ValueError(f"Unknown existing solution pool: {instance_id}")
        if entry.get("instance_sha256") != instance_entry["sha256"]:
            raise ValueError(f"Stale instance checksum for pool: {instance_id}")
        if entry.get("instance_path") != instance_entry["path"]:
            raise ValueError(f"Stale instance path for pool: {instance_id}")
        if entry.get("partition") != instance_entry["partition"]:
            raise ValueError(f"Stale partition for pool: {instance_id}")
        pool_path_value = entry["pool_path"]
        if pool_path_value in seen_paths:
            raise ValueError("Existing solution-pool manifest contains duplicate paths.")
        seen_paths.add(pool_path_value)
        pool_path = root / pool_path_value
        if not pool_path.exists():
            raise ValueError(f"Missing existing solution pool: {pool_path}")
        if _sha256(pool_path) != entry["pool_sha256"]:
            raise ValueError(f"Solution-pool checksum mismatch: {pool_path}")
        indexed[instance_id] = entry
    if existing_manifest.get("labeled_instance_count") != len(indexed):
        raise ValueError("Existing solution-pool manifest count is inconsistent.")
    return indexed


def label_dataset(
    dataset_root: str | Path,
    config: SolutionPoolConfig | None = None,
    *,
    partitions: Iterable[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Incrementally generate verified pools without mutating dataset instances.

    Existing entries are checksum- and evaluator-audited before they are
    skipped. Every newly completed pool is atomically merged into the manifest,
    so an interrupted labeling run can safely resume.
    """

    config = config or SolutionPoolConfig()
    root = Path(dataset_root)
    dataset_manifest = load_manifest(root / "manifest.json")
    selected_partitions = None if partitions is None else set(partitions)
    if selected_partitions is not None:
        unknown = selected_partitions - set(dataset_manifest["partitions"])
        if unknown:
            raise ValueError(f"Unknown dataset partitions: {sorted(unknown)}")
    entries = [
        entry
        for entry in dataset_manifest["instances"]
        if selected_partitions is None or entry["partition"] in selected_partitions
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive when specified.")
        entries = entries[:limit]
    if not entries:
        raise ValueError("No dataset instances were selected for solution-pool labeling.")

    existing_manifest = _load_existing_manifest(root, dataset_manifest, config)
    if existing_manifest is not None and not resume:
        raise FileExistsError(
            "solution_pool_manifest.json already exists; enable resume or remove it."
        )
    entries_by_instance = _index_existing_entries(
        root, dataset_manifest, existing_manifest
    )
    for entry in entries:
        existing = entries_by_instance.get(entry["instance_id"])
        if existing is not None and not force:
            _audit_existing_entry(root, entry, existing)
            continue
        instance = load_manifest_instance(root, entry)
        pool = build_solution_pool(instance, config)
        audit_solution_pool(instance, pool)
        relative_path = (
            Path("solution_pools") / entry["partition"] / f"{instance.instance_id}.npz"
        )
        saved = save_solution_pool(pool, root / relative_path)
        entries_by_instance[instance.instance_id] = _pool_entry(
            root,
            entry,
            instance,
            pool,
            saved,
            relative_path,
        )
        _write_merged_manifest(root, dataset_manifest, config, entries_by_instance)

    return _write_merged_manifest(root, dataset_manifest, config, entries_by_instance)


def audit_solution_pool_manifest(dataset_root: str | Path) -> dict[str, Any]:
    """Verify pool checksums, instance checksums, and all stored labels."""

    root = Path(dataset_root)
    dataset_manifest = load_manifest(root / "manifest.json")
    instance_entries: Mapping[str, Mapping[str, Any]] = {
        entry["instance_id"]: entry for entry in dataset_manifest["instances"]
    }
    with (root / "solution_pool_manifest.json").open("r", encoding="utf-8") as stream:
        pool_manifest = json.load(stream)
    if pool_manifest.get("schema_version") != SOLUTION_POOL_SCHEMA_VERSION:
        raise ValueError("Unsupported solution-pool manifest schema version.")
    if pool_manifest.get("dataset_name") != dataset_manifest["dataset_name"]:
        raise ValueError("Solution-pool manifest refers to a different dataset.")
    if pool_manifest.get("labeled_instance_count") != len(pool_manifest.get("pools", [])):
        raise ValueError("Solution-pool manifest count is inconsistent.")
    seen: set[str] = set()
    seen_paths: set[str] = set()
    for pool_entry in pool_manifest["pools"]:
        instance_id = pool_entry["instance_id"]
        if instance_id in seen:
            raise ValueError("Solution-pool manifest contains duplicate instance IDs.")
        seen.add(instance_id)
        instance_entry = instance_entries.get(instance_id)
        if instance_entry is None:
            raise ValueError(f"Unknown instance in solution-pool manifest: {instance_id}")
        if pool_entry["instance_sha256"] != instance_entry["sha256"]:
            raise ValueError(f"Stale instance checksum for pool: {instance_id}")
        if pool_entry["instance_path"] != instance_entry["path"]:
            raise ValueError(f"Stale instance path for pool: {instance_id}")
        if pool_entry["partition"] != instance_entry["partition"]:
            raise ValueError(f"Stale partition for pool: {instance_id}")
        if pool_entry["pool_path"] in seen_paths:
            raise ValueError("Solution-pool manifest contains duplicate pool paths.")
        seen_paths.add(pool_entry["pool_path"])
        pool_path = root / pool_entry["pool_path"]
        if _sha256(pool_path) != pool_entry["pool_sha256"]:
            raise ValueError(f"Solution-pool checksum mismatch: {pool_path}")
        instance = load_manifest_instance(root, instance_entry)
        pool = load_solution_pool(pool_path)
        audit_solution_pool(instance, pool)
        if pool.size != pool_entry["pool_size"]:
            raise ValueError(f"Solution-pool size mismatch: {instance_id}")
        if not np.isclose(pool.latencies.min(), pool_entry["minimum_latency"]):
            raise ValueError(f"Solution-pool minimum latency mismatch: {instance_id}")
        if not np.isclose(pool.latencies.max(), pool_entry["maximum_latency"]):
            raise ValueError(f"Solution-pool maximum latency mismatch: {instance_id}")
        if pool.metadata["termination_reason"] != pool_entry["termination_reason"]:
            raise ValueError(f"Solution-pool termination mismatch: {instance_id}")
    return pool_manifest
