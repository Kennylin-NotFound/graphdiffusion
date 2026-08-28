"""Export standard paper-ready figures from a completed experiment run."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments import export_run_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_run_figures(args.run_directory, output_directory=args.output)
    print(
        f"run={payload['run_directory']} figures={','.join(payload['figures'])}"
    )


if __name__ == "__main__":
    main()
