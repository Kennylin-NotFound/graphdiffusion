"""Run Stage 3 checkpoint-only calibration and optionally freeze its result."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.phase6ee_stage3 import (
    finalize_stage3_calibration,
    run_stage3_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = root / "artifacts" / "phase6e-e-stage3" / "pilot_preparation_lock.json"
    if args.finalize:
        if args.smoke or args.limit is not None:
            raise ValueError("Finalization cannot be combined with smoke/limit.")
        print(finalize_stage3_calibration(lock, implementation_root=root))
        return
    print(
        run_stage3_calibration(
            lock,
            implementation_root=root,
            limit=args.limit,
            smoke=args.smoke,
        )
    )


if __name__ == "__main__":
    main()
