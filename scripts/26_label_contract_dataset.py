"""Resume solution-pool labeling using the immutable dataset contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.data import load_manifest
from gdm_factor_diffusion.solver import SolutionPoolConfig
from gdm_factor_diffusion.solver.dataset_labeling import label_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--partition", action="append", dest="partitions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--solver-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    manifest = load_manifest(root / "manifest.json")
    labeling = manifest["generation_config"].get("contract", {}).get("labeling")
    if labeling is None:
        raise ValueError("Dataset manifest does not define a labeling contract.")
    config = SolutionPoolConfig(
        target_size=int(labeling["target_size"]),
        beta=float(labeling["beta"]),
        total_time_limit_seconds=float(labeling["total_time_limit_seconds"]),
        mip_gap=float(labeling["mip_gap"]),
        threads=int(labeling["threads"]),
        seed=int(labeling["seed"]),
        output_flag=args.solver_output,
    )
    result = label_dataset(
        root,
        config,
        partitions=args.partitions,
        limit=args.limit,
        resume=True,
        force=args.force,
    )
    print(
        f"dataset={result['dataset_name']} "
        f"labeled={result['labeled_instance_count']}/"
        f"{result['dataset_instance_count']} root={root}"
    )


if __name__ == "__main__":
    main()
