"""Run or resume selected Phase 6E-B smoke/full training entries."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.phase6eb_campaign import (
    run_phase6eb_training,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    index = run_phase6eb_training(
        root / "artifacts" / "phase6e-b-training" / "campaign_lock.json",
        implementation_root=root,
        mode=args.mode,
        variants=args.variant,
        seeds=args.seed,
    )
    print(f"mode={args.mode} completed_runs={len(index['runs'])}")
