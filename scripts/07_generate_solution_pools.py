"""Generate verified Phase 1C solution pools for a dataset manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.solver import SolutionPoolConfig
from gdm_factor_diffusion.solver.dataset_labeling import label_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=None,
        help="Dataset root containing manifest.json.",
    )
    parser.add_argument("--target-size", type=int, default=8)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--partition", action="append", dest="partitions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--solver-output", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate selected pools after validating manifest compatibility.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Refuse to run when a solution-pool manifest already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    root = args.dataset_root or (
        implementation_root / "artifacts" / "datasets" / "phase1b-smoke"
    )
    config = SolutionPoolConfig(
        target_size=args.target_size,
        beta=args.beta,
        total_time_limit_seconds=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.seed,
        output_flag=args.solver_output,
    )
    manifest = label_dataset(
        root,
        config,
        partitions=args.partitions,
        limit=args.limit,
        resume=not args.no_resume,
        force=args.force,
    )
    print(
        f"dataset={manifest['dataset_name']} "
        f"labeled={manifest['labeled_instance_count']} root={root}"
    )
    for entry in manifest["pools"]:
        print(
            f"{entry['instance_id']}: size={entry['pool_size']} "
            f"latency=[{entry['minimum_latency']:.8g},{entry['maximum_latency']:.8g}] "
            f"termination={entry['termination_reason']} "
            f"elapsed={entry['elapsed_seconds']:.3f}s"
        )


if __name__ == "__main__":
    main()
