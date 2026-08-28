"""Run or resume selected groups from the locked Phase 6D-C campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.sealed_campaign import run_sealed_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("artifacts/phase6d-c-final/campaign_lock.json"),
    )
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        help="Repeatable group such as main-stochastic or scale-optimization.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = args.lock if args.lock.is_absolute() else root / args.lock
    index = run_sealed_campaign(lock, implementation_root=root, groups=args.group)
    print(f"completed_runs={len(index['runs'])} index={root / 'artifacts/phase6d-c-final/run_index.json'}")


if __name__ == "__main__":
    main()
