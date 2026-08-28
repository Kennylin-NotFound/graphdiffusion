"""Lock the validation-only Phase 6E-E Stage 1 diagnostics."""

from pathlib import Path

from gdm_factor_diffusion.experiments import prepare_phase6ee_stage1


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lock = prepare_phase6ee_stage1(
        root / "configs" / "phase6e_e_stage1_diagnostics.yaml",
        implementation_root=root,
    )
    print(
        f"scope={lock['scope']} seeds={len(lock['seeds'])} "
        f"instances={len(lock['instance_ids'])}"
    )

