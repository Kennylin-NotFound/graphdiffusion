from pathlib import Path

from gdm_factor_diffusion.experiments import finalize_phase6ee_stage2b


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result = finalize_phase6ee_stage2b(
        ROOT / "artifacts" / "phase6e-e-stage2b" / "stage2b_lock.json",
        implementation_root=ROOT,
    )
    print(result["interpretation"])
