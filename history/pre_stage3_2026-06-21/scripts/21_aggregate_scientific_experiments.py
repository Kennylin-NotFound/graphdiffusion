"""Aggregate multiple completed Phase 6 experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gdm_factor_diffusion.experiments import aggregate_run_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = aggregate_run_directories(args.runs, output=args.output)
    print(
        f"runs={payload['runs']} unique_seeds={payload['unique_seeds']} "
        f"scope={payload['aggregation_scope']} coverage={payload['instances_per_run']} "
        f"methods={','.join(payload['methods'])} output={args.output}"
    )
    print(json.dumps(payload["methods"], indent=2))


if __name__ == "__main__":
    main()
