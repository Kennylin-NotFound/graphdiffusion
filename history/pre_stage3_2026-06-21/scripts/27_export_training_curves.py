"""Export single-seed production-training diagnostic curves."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments import export_training_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = export_training_curves(
        args.run_directory,
        output_directory=args.output,
    )
    print(f"figures={','.join(payload['figures'])}")


if __name__ == "__main__":
    main()
