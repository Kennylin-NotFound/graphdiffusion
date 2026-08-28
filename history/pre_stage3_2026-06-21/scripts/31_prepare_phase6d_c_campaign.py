"""Generate and lock the sealed Phase 6D-C experiment manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments.sealed_campaign import prepare_sealed_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase6d_c_sealed_campaign.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    lock = prepare_sealed_campaign(config, implementation_root=root)
    print(
        f"scope={lock['scope']} manifests={len(lock['manifests'])} "
        f"datasets={','.join(lock['datasets'])}"
    )


if __name__ == "__main__":
    main()
