"""Verify, aggregate, and freeze a completed Phase 6E-A campaign."""

from pathlib import Path

from gdm_factor_diffusion.experiments.phase6e_campaign import finalize_phase6e_a_campaign


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    evidence = finalize_phase6e_a_campaign(
        root / "artifacts" / "phase6e-a-inference" / "campaign_lock.json",
        implementation_root=root,
    )
    print(f"scope={evidence['scope']} runs={len(evidence['runs'])}")
