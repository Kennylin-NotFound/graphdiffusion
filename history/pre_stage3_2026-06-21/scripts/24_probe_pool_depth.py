"""Measure deeper solution-pool generation on selected hard scale instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.data import load_manifest, load_manifest_instance
from gdm_factor_diffusion.solver import SolutionPoolConfig, build_solution_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("instance_ids", nargs="+")
    parser.add_argument("--target-size", type=int, default=16)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.dataset_root / "manifest.json")
    by_id = {entry["instance_id"]: entry for entry in manifest["instances"]}
    records = []
    for instance_id in args.instance_ids:
        instance = load_manifest_instance(args.dataset_root, by_id[instance_id])
        pool = build_solution_pool(
            instance,
            SolutionPoolConfig(
                target_size=args.target_size,
                beta=5.0,
                total_time_limit_seconds=args.time_limit,
                mip_gap=0.0,
                threads=1,
                seed=0,
                output_flag=False,
            ),
        )
        record = {
            "instance_id": instance_id,
            "num_services": instance.num_services,
            "num_devices": instance.num_devices,
            "candidate_edges": int(instance.compatibility_mask.sum()),
            "requested_size": args.target_size,
            "actual_size": pool.size,
            "elapsed_seconds": pool.metadata["elapsed_seconds"],
            "termination_reason": pool.metadata["termination_reason"],
            "minimum_latency": float(pool.latencies[0]),
            "maximum_latency": float(pool.latencies[-1]),
        }
        records.append(record)
        print(
            f"instance={instance_id} size={pool.size} "
            f"termination={pool.metadata['termination_reason']} "
            f"elapsed={pool.metadata['elapsed_seconds']:.3f}s"
        )
    write_json(
        args.output,
        {
            "schema_version": "1.0",
            "dataset_root": str(args.dataset_root.resolve()),
            "target_size": args.target_size,
            "time_limit_seconds": args.time_limit,
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
