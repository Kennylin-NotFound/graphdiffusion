"""Prepare, run, and freeze the Stage 3.6 checkpoint-only K calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.phase6ee_stage36 import (
    finalize_stage36_efficiency,
    prepare_stage36_efficiency,
    run_stage36_efficiency,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "finalize"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "phase6e_e_stage36_efficiency.yaml"
    lock = root / "artifacts" / "phase6e-e-stage36" / "efficiency_preparation_lock.json"
    if args.action == "prepare":
        if args.smoke or args.limit is not None:
            raise ValueError("Prepare does not accept smoke/limit.")
        print(prepare_stage36_efficiency(config, implementation_root=root))
    elif args.action == "run":
        print(
            run_stage36_efficiency(
                lock,
                implementation_root=root,
                limit=args.limit,
                smoke=args.smoke,
            )
        )
    else:
        if args.smoke or args.limit is not None:
            raise ValueError("Finalize does not accept smoke/limit.")
        print(finalize_stage36_efficiency(lock, implementation_root=root))


if __name__ == "__main__":
    main()

