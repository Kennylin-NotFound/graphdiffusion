"""Aggregate and freeze Phase 6E-B final-ID evidence."""

from pathlib import Path

from gdm_factor_diffusion.experiments.phase6eb_campaign import (
    finalize_phase6eb_evaluation,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    evidence = finalize_phase6eb_evaluation(
        root / "artifacts" / "phase6e-b-evaluation" / "campaign_lock.json",
        implementation_root=root,
    )
    print(f"scope={evidence['scope']} runs={len(evidence['runs'])}")
