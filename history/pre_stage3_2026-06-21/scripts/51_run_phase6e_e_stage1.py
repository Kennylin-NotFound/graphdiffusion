"""Run or resume Phase 6E-E Stage 1 diagnostics."""

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments import run_phase6ee_stage1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="append", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    index = run_phase6ee_stage1(
        root / "artifacts" / "phase6e-e-stage1" / "stage1_lock.json",
        implementation_root=root,
        seeds=args.seed,
    )
    print(f"completed_runs={len(index['runs'])}")

