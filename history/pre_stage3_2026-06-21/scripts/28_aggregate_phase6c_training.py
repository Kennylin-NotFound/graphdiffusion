"""Aggregate a fixed independent-seed training campaign and freeze checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.experiments import aggregate_and_freeze_training_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("configs/phase6c_five_seed_campaign.yaml"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    campaign_path = (
        args.campaign
        if args.campaign.is_absolute()
        else implementation_root / args.campaign
    )
    campaign = load_config(campaign_path)["campaign"]
    output = args.output or implementation_root / campaign["output_root"]
    payload = aggregate_and_freeze_training_runs(
        args.run_directories,
        expected_seeds=campaign["seeds"],
        expected_steps=int(campaign["expected_steps"]),
        output_directory=output,
    )
    best = payload["best_checkpoint_summary"]
    print(
        f"seeds={len(payload['seeds'])} "
        f"best_step_mean={best['best_step']['mean']:.1f} "
        f"gap_mean={best['mean_gap_to_pool_best']['mean']:.6f} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
