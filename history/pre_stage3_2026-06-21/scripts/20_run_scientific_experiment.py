"""Run one validated Phase 6 scientific experiment manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gdm_factor_diffusion.experiments import (
    evaluate_experiment,
    load_experiment_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    manifest = load_experiment_manifest(args.manifest)
    run_directory = evaluate_experiment(
        manifest,
        implementation_root=implementation_root,
        output_root=args.output_root,
    )
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    print(
        f"run={run_directory} instances={summary['instances']} "
        f"methods={summary['methods']} quality={summary['quality_fingerprint']} "
        f"success_rate={summary['final_success_rate']:.6f} "
        f"successful_outputs_verified={summary['all_successful_outputs_verified']}"
    )


if __name__ == "__main__":
    main()
