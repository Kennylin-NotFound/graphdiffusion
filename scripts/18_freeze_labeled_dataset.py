"""Fully audit and freeze the core manifests of a labeled dataset."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.data import (
    audit_graph_readiness,
    load_manifest,
    load_manifest_instance,
)
from gdm_factor_diffusion.solver import verify_placement
from gdm_factor_diffusion.solver.dataset_labeling import audit_solution_pool_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    manifest = load_manifest(root / "manifest.json")
    pool_manifest = audit_solution_pool_manifest(root)
    if pool_manifest["labeled_instance_count"] != manifest["instance_count"]:
        raise ValueError("Cannot freeze a partially labeled dataset.")
    contract_labeling = manifest["generation_config"].get("contract", {}).get(
        "labeling"
    )
    if contract_labeling is not None:
        for key, expected in contract_labeling.items():
            if pool_manifest["config"].get(key) != expected:
                raise ValueError(
                    f"Labeling config disagrees with dataset contract for {key!r}."
                )

    pools_by_instance = {
        entry["instance_id"]: entry for entry in pool_manifest["pools"]
    }
    by_partition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest["instances"]:
        instance = load_manifest_instance(root, entry)
        witness = np.asarray(entry["witness_placement"], dtype=np.int64)
        if not verify_placement(instance, witness).feasible:
            raise ValueError(f"Generation witness is infeasible: {instance.instance_id}")
        if not audit_graph_readiness(instance, witness).ready:
            raise ValueError(f"Instance is not graph-ready: {instance.instance_id}")
        pool_entry = pools_by_instance[instance.instance_id]
        by_partition[entry["partition"]].append(
            {
                "num_services": entry["num_services"],
                "num_devices": entry["num_devices"],
                "num_dependencies": entry["num_dependencies"],
                "candidate_edges": entry["graph_readiness"]["relation_counts"][
                    "service_to_device_candidates"
                ],
                "pool_size": pool_entry["pool_size"],
                "pool_spread": (
                    pool_entry["maximum_latency"] / pool_entry["minimum_latency"] - 1.0
                ),
            }
        )

    partition_stats = {}
    for partition, entries in by_partition.items():
        partition_stats[partition] = {
            "instances": len(entries),
            "services_min": min(entry["num_services"] for entry in entries),
            "services_max": max(entry["num_services"] for entry in entries),
            "services_mean": mean(entry["num_services"] for entry in entries),
            "devices_min": min(entry["num_devices"] for entry in entries),
            "devices_max": max(entry["num_devices"] for entry in entries),
            "dependencies_min": min(entry["num_dependencies"] for entry in entries),
            "dependencies_max": max(entry["num_dependencies"] for entry in entries),
            "candidate_edges_min": min(entry["candidate_edges"] for entry in entries),
            "candidate_edges_max": max(entry["candidate_edges"] for entry in entries),
            "pool_size_min": min(entry["pool_size"] for entry in entries),
            "pool_size_max": max(entry["pool_size"] for entry in entries),
            "pool_size_mean": mean(entry["pool_size"] for entry in entries),
            "pool_spread_mean": mean(entry["pool_spread"] for entry in entries),
        }

    core_files = ("catalog.json", "manifest.json", "solution_pool_manifest.json")
    freeze = {
        "schema_version": "1.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": manifest["dataset_name"],
        "dataset_instance_count": manifest["instance_count"],
        "labeled_instance_count": pool_manifest["labeled_instance_count"],
        "verified_solution_count": sum(
            entry["pool_size"] for entry in pool_manifest["pools"]
        ),
        "core_sha256": {name: _sha256(root / name) for name in core_files},
        "labeling_config": pool_manifest["config"],
        "termination_reasons": dict(
            sorted(
                Counter(
                    entry["termination_reason"] for entry in pool_manifest["pools"]
                ).items()
            )
        ),
        "partition_stats": partition_stats,
        "audit_contracts": [
            "dataset manifest and split contract",
            "instance checksums and generation witnesses",
            "graph-readiness relations",
            "solution-pool checksums and uniqueness",
            "hard placement feasibility",
            "exact latency and energy distribution",
        ],
    }
    destination = write_json(root / "dataset_freeze.json", freeze)
    print(
        f"dataset={freeze['dataset_name']} instances={freeze['dataset_instance_count']} "
        f"solutions={freeze['verified_solution_count']} freeze={destination}"
    )


if __name__ == "__main__":
    main()
