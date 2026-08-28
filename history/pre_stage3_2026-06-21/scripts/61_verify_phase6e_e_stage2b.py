from pathlib import Path

from gdm_factor_diffusion.experiments import verify_phase6ee_stage2b_evidence


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result = verify_phase6ee_stage2b_evidence(
        ROOT / "artifacts" / "phase6e-e-stage2b" / "final_evidence_freeze.json",
        implementation_root=ROOT,
    )
    print(
        {
            "record_count": result["record_count"],
            "selected_variant": result["selected_variant"],
            "interpretation": result["interpretation"]["conclusion"],
        }
    )
