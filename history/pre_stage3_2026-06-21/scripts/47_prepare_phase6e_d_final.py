"""Generate final-ID manifests after the Phase 6E-D budget freeze."""

from pathlib import Path

from gdm_factor_diffusion.experiments import prepare_phase6ed_final


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lock = prepare_phase6ed_final(
        root / "artifacts" / "phase6e-d-time-matched" / "budget_freeze.json",
        implementation_root=root,
    )
    print(
        f"scope={lock['scope']} selected_direct_k={lock['selected_direct_k']} "
        f"manifests={len(lock['manifests'])}"
    )

