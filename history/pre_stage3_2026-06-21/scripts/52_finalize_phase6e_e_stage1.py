"""Aggregate Phase 6E-E Stage 1 diagnostics and issue Gate R1."""

from pathlib import Path

from gdm_factor_diffusion.experiments import finalize_phase6ee_stage1


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    evidence = finalize_phase6ee_stage1(
        root / "artifacts" / "phase6e-e-stage1" / "stage1_lock.json",
        implementation_root=root,
    )
    gate = evidence["aggregate"]["gate_r1"]
    print(f"recommendation={gate['recommendation']}")

