"""Run the initial or conditional extension Phase 6E-D validation grid."""

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments import run_phase6ed_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", choices=("initial", "extension"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    index = run_phase6ed_validation(
        root / "artifacts" / "phase6e-d-time-matched" / "validation_lock.json",
        implementation_root=root,
        groups=args.group,
    )
    print(f"completed_runs={len(index['runs'])}")

