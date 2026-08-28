import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments import run_phase6ee_stage2b


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Run one locked seed; omit to run or resume all five seeds.",
    )
    args = parser.parse_args()
    result = run_phase6ee_stage2b(
        ROOT / "artifacts" / "phase6e-e-stage2b" / "stage2b_lock.json",
        implementation_root=ROOT,
        seeds=args.seed,
    )
    print(result)
