"""Generate final-ID manifests from frozen Phase 6E-B checkpoints."""

from pathlib import Path

from gdm_factor_diffusion.experiments.phase6eb_campaign import (
    prepare_phase6eb_evaluation,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lock = prepare_phase6eb_evaluation(
        root
        / "artifacts"
        / "phase6e-b-training"
        / "training_evidence_freeze.json",
        implementation_root=root,
    )
    print(f"scope={lock['scope']} manifests={len(lock['manifests'])}")
