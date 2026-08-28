"""Generate and lock the Phase 6E-D validation calibration."""

from pathlib import Path

from gdm_factor_diffusion.experiments import prepare_phase6ed_validation


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lock = prepare_phase6ed_validation(
        root / "configs" / "phase6e_d_time_matched_campaign.yaml",
        implementation_root=root,
    )
    print(f"scope={lock['scope']} manifests={len(lock['manifests'])}")

