"""Freeze the Stage 3 development and training contract before data creation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from gdm_factor_diffusion.experiments.phase6ee_stage3 import prepare_stage3_contract


def main() -> None:
    implementation_root = Path(__file__).resolve().parents[1]
    lock = prepare_stage3_contract(implementation_root)
    path = (
        implementation_root
        / "artifacts"
        / "phase6e-e-stage3"
        / "stage3_contract_lock.json"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    print(
        f"phase={lock['phase']} instances={lock['development_instance_count']} "
        f"training_seed={lock['training_seed']} final_data_exists=False"
    )
    print(f"lock={path} sha256={digest}")


if __name__ == "__main__":
    main()
