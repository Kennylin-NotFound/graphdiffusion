"""Prepare the hash-locked Stage 3 calibration contract without opening pilot."""

from pathlib import Path

from gdm_factor_diffusion.experiments.phase6ee_stage3 import prepare_stage3_pilot


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lock = prepare_stage3_pilot(
        root / "configs" / "phase6e_e_stage3_pilot.yaml",
        implementation_root=root,
    )
    print(
        f"scope={lock['scope']} checkpoint_ids={len(lock['checkpoint_instance_ids'])} "
        f"pilot_ids={len(lock['pilot_instance_ids'])}"
    )

