import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments import run_phase6ee_stage2a


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    result = run_phase6ee_stage2a(
        ROOT / "artifacts" / "phase6e-e-stage2a" / "stage2a_lock.json",
        implementation_root=ROOT,
        split="confirmation",
        seeds=args.seed,
        calibration_freeze_path=(
            ROOT / "artifacts" / "phase6e-e-stage2a" / "calibration_freeze.json"
        ),
    )
    print(result)
