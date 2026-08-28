"""Export Phase 6B scaling figures from a completed probe."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.scale_probe import export_scale_probe_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = export_scale_probe_figures(
        args.run_directory,
        output_directory=args.output,
    )
    print(f"figures={','.join(payload['figures'])}")


if __name__ == "__main__":
    main()
