"""Audit a generated dataset manifest, instances, witnesses, and graph inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from gdm_factor_diffusion.data import (
    audit_graph_readiness,
    load_manifest,
    load_manifest_instance,
)
from gdm_factor_diffusion.solver import verify_placement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=None,
        help="Dataset root containing manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    root = args.dataset_root or (
        implementation_root / "artifacts" / "datasets" / "phase1b-smoke"
    )
    manifest = load_manifest(root / "manifest.json")
    by_partition: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest["instances"]:
        instance = load_manifest_instance(root, entry)
        witness = np.asarray(entry["witness_placement"], dtype=np.int64)
        if not verify_placement(instance, witness).feasible:
            raise RuntimeError(f"Witness failed verification: {entry['instance_id']}")
        audit = audit_graph_readiness(instance, witness)
        if not audit.ready:
            raise RuntimeError(f"Graph audit failed: {entry['instance_id']}")
        by_partition[entry["partition"]].append(entry)

    print(f"dataset={manifest['dataset_name']} instances={manifest['instance_count']}")
    for partition, entries in by_partition.items():
        services = [entry["num_services"] for entry in entries]
        dependencies = [entry["num_dependencies"] for entry in entries]
        candidates = [
            entry["graph_readiness"]["relation_counts"][
                "service_to_device_candidates"
            ]
            for entry in entries
        ]
        print(
            f"{partition}: n={len(entries)} "
            f"services=[{min(services)},{max(services)}] "
            f"dependencies=[{min(dependencies)},{max(dependencies)}] "
            f"candidate_edges=[{min(candidates)},{max(candidates)}]"
        )


if __name__ == "__main__":
    main()
