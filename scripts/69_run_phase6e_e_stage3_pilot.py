"""Prepare, run, resume, or finalize the one-time Stage 3 pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.phase6ee_stage3_pilot import (
    finalize_stage3_pilot,
    prepare_stage3_pilot_execution,
    run_stage3_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    contract = root / "artifacts" / "phase6e-e-stage3" / "pilot_contract_lock.json"
    execution = root / "artifacts" / "phase6e-e-stage3" / "pilot_execution_lock.json"
    if args.action == "prepare":
        print(prepare_stage3_pilot_execution(contract, implementation_root=root))
    elif args.action == "run":
        print(run_stage3_pilot(execution, implementation_root=root))
    else:
        print(finalize_stage3_pilot(execution, implementation_root=root))


if __name__ == "__main__":
    main()

