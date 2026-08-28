"""Run the Stage 3.7 optimized diffusion/direct development gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.phase6ee_stage37 import (
    finalize_stage37,
    prepare_stage37,
    run_stage37_calibration,
    run_stage37_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "run", "finalize"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "phase6e_e_stage37_matched_time.yaml"
    lock = root / "artifacts" / "phase6e-e-stage37" / "preparation_lock.json"
    if args.action == "prepare":
        print(prepare_stage37(config, implementation_root=root))
    elif args.action == "preflight":
        print(run_stage37_preflight(lock, implementation_root=root))
    elif args.action == "run":
        print(
            run_stage37_calibration(
                lock,
                implementation_root=root,
                limit=args.limit,
                smoke=args.smoke,
            )
        )
    else:
        print(finalize_stage37(lock, implementation_root=root))


if __name__ == "__main__":
    main()

