"""Manage the Stage 3.8 sealed multi-seed confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.phase6ee_stage38 import (
    authorize_sealed_data,
    finalize_stage38,
    finalize_stage38_training,
    prepare_stage38,
    run_stage38_evaluation,
    verify_training_open,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "verify-training", "finalize-training", "authorize-data", "run", "finalize"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "phase6e_e_stage38_sealed.yaml"
    lock = root / "artifacts" / "phase6e-e-stage38" / "preparation_lock.json"
    if args.action == "prepare":
        print(prepare_stage38(config, implementation_root=root))
    elif args.action == "verify-training":
        print(verify_training_open(lock, implementation_root=root))
    elif args.action == "finalize-training":
        print(finalize_stage38_training(lock, implementation_root=root))
    elif args.action == "authorize-data":
        print(authorize_sealed_data(lock, implementation_root=root))
    elif args.action == "run":
        print(run_stage38_evaluation(lock, implementation_root=root))
    else:
        print(finalize_stage38(lock, implementation_root=root))


if __name__ == "__main__":
    main()
