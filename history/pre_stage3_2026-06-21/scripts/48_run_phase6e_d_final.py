"""Run the locked Phase 6E-D final-ID comparison."""

from pathlib import Path

from gdm_factor_diffusion.experiments import run_phase6ed_final


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    index = run_phase6ed_final(
        root / "artifacts" / "phase6e-d-time-matched" / "final_lock.json",
        implementation_root=root,
    )
    print(f"completed_runs={len(index['runs'])}")

