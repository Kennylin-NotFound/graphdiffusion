from pathlib import Path

from gdm_factor_diffusion.experiments import finalize_phase6ee_stage2a_confirmation


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result = finalize_phase6ee_stage2a_confirmation(
        ROOT / "artifacts" / "phase6e-e-stage2a" / "stage2a_lock.json",
        ROOT / "artifacts" / "phase6e-e-stage2a" / "calibration_freeze.json",
        implementation_root=ROOT,
    )
    print(result["gate_r2"])
