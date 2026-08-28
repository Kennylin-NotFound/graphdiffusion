"""Verify the frozen five-seed checkpoint and training-evidence hashes."""

from __future__ import annotations

import argparse
from pathlib import Path

from gdm_factor_diffusion.experiments import verify_checkpoint_freeze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("freeze", type=Path)
    args = parser.parse_args()
    payload = verify_checkpoint_freeze(args.freeze)
    print(
        f"seeds={len(payload['seeds'])} "
        f"steps={payload['expected_steps']} freeze={args.freeze.resolve()}"
    )


if __name__ == "__main__":
    main()
