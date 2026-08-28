from pathlib import Path

from gdm_factor_diffusion.experiments import finalize_phase6ee_stage2a_calibration


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result = finalize_phase6ee_stage2a_calibration(
        ROOT / "artifacts" / "phase6e-e-stage2a" / "stage2a_lock.json",
        implementation_root=ROOT,
    )
    print(result["selected_variant"])
