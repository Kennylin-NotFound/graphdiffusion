"""Run selected locked Phase 6E-A groups with resumable artifact indexing."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.phase6e_campaign import run_phase6e_a_campaign


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    index = run_phase6e_a_campaign(
        root / "artifacts" / "phase6e-a-inference" / "campaign_lock.json",
        implementation_root=root,
        groups=args.group,
    )
    print(f"completed_runs={len(index['runs'])}")
