"""Verify, aggregate, and freeze all completed Phase 6D-C evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.sealed_campaign import finalize_sealed_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("artifacts/phase6d-c-final/campaign_lock.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = args.lock if args.lock.is_absolute() else root / args.lock
    evidence = finalize_sealed_campaign(lock, implementation_root=root)
    print(
        f"scope={evidence['scope']} runs={len(evidence['runs'])} "
        f"freeze={root / 'artifacts/phase6d-c-final/final_evidence_freeze.json'}"
    )


if __name__ == "__main__":
    main()
