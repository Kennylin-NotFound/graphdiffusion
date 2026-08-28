"""Freeze three checkpoints per Phase 6E-B training variant."""

from pathlib import Path

from gdm_factor_diffusion.experiments.phase6eb_campaign import (
    finalize_phase6eb_training,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    evidence = finalize_phase6eb_training(
        root / "artifacts" / "phase6e-b-training" / "campaign_lock.json",
        implementation_root=root,
    )
    print(
        f"scope={evidence['scope']} "
        f"variant_freezes={len(evidence['variant_freezes'])}"
    )
