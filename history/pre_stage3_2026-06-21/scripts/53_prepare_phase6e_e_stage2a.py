from pathlib import Path

from gdm_factor_diffusion.experiments import prepare_phase6ee_stage2a


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result = prepare_phase6ee_stage2a(
        ROOT / "configs" / "phase6e_e_stage2a_rescue.yaml",
        implementation_root=ROOT,
    )
    print(ROOT / result["output_root"] / "stage2a_lock.json")
