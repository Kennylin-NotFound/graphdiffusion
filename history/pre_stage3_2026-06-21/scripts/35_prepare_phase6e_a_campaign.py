"""Generate and lock the Phase 6E-A inference-only ablation campaign."""

from pathlib import Path

from gdm_factor_diffusion.experiments.phase6e_campaign import prepare_phase6e_a_campaign


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lock = prepare_phase6e_a_campaign(
        root / "configs" / "phase6e_a_inference_campaign.yaml",
        implementation_root=root,
    )
    print(f"scope={lock['scope']} manifests={len(lock['manifests'])}")
