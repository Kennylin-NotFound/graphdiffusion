"""Aggregate, report, and freeze Phase 6E-D final evidence."""

from pathlib import Path

from gdm_factor_diffusion.experiments import finalize_phase6ed_final


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    evidence = finalize_phase6ed_final(
        root / "artifacts" / "phase6e-d-time-matched" / "final_lock.json",
        implementation_root=root,
    )
    print(
        f"scope={evidence['scope']} interpretation={evidence['interpretation']}"
    )
