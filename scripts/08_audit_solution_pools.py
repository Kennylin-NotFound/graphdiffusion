"""Audit saved solution pools against immutable instances and exact evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from gdm_factor_diffusion.solver.dataset_labeling import audit_solution_pool_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=None,
        help="Dataset root containing solution_pool_manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    root = args.dataset_root or (
        implementation_root / "artifacts" / "datasets" / "phase1b-smoke"
    )
    manifest = audit_solution_pool_manifest(root)
    sizes = [entry["pool_size"] for entry in manifest["pools"]]
    termination = Counter(entry["termination_reason"] for entry in manifest["pools"])
    print(
        f"dataset={manifest['dataset_name']} "
        f"audited={manifest['labeled_instance_count']} "
        f"pool_sizes=[{min(sizes)},{max(sizes)}] "
        f"termination={dict(termination)}"
    )


if __name__ == "__main__":
    main()
