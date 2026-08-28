"""Run or reuse the locked three-seed Phase 6E-B final-ID evaluation."""

from pathlib import Path

from gdm_factor_diffusion.experiments.phase6eb_campaign import (
    run_phase6eb_evaluation,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    index = run_phase6eb_evaluation(
        root / "artifacts" / "phase6e-b-evaluation" / "campaign_lock.json",
        implementation_root=root,
    )
    print(f"completed_runs={len(index['runs'])}")
