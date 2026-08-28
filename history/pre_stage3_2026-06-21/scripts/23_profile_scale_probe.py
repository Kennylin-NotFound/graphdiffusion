"""Profile a frozen Phase 6B scale-probe dataset end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gdm_factor_diffusion.experiments.scale_probe import (
    ScaleProbeConfig,
    run_scale_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--milp-time-limit", type=float, default=15.0)
    parser.add_argument("--training-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    output = args.output_root or implementation_root / "artifacts" / "phase6b-probes"
    run = run_scale_probe(
        args.dataset_root,
        args.checkpoint,
        output_root=output,
        config=ScaleProbeConfig(
            seed=args.seed,
            device=args.device,
            milp_time_limit_seconds=args.milp_time_limit,
            training_profile_steps=args.training_steps,
        ),
        project_root=implementation_root,
    )
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    print(
        f"run={run} records={summary['records']} "
        f"verified={summary['all_learned_outputs_verified']}"
    )


if __name__ == "__main__":
    main()
