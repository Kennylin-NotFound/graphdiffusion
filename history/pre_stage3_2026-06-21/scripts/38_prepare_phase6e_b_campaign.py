"""Generate and lock Phase 6E-B retraining configurations."""

from pathlib import Path

from gdm_factor_diffusion.experiments.phase6eb_campaign import (
    prepare_phase6eb_campaign,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    lock = prepare_phase6eb_campaign(
        root / "configs" / "phase6e_b_training_campaign.yaml",
        implementation_root=root,
    )
    full = sum(entry["mode"] == "full" for entry in lock["training_configs"])
    smoke = sum(entry["mode"] == "smoke" for entry in lock["training_configs"])
    print(f"scope={lock['scope']} full_configs={full} smoke_configs={smoke}")
