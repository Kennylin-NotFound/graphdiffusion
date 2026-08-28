"""Freeze the complete Stage 3 implementation and data state before training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gdm_factor_diffusion.common.logging import write_json


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_root = root / "artifacts" / "datasets" / "phase6e-e-stage3-development"
    stage3_root = root / "artifacts" / "phase6e-e-stage3"
    mvp_root = root / "artifacts" / "phase6e-e-stage3-pretraining"
    contract = _read(stage3_root / "stage3_contract_lock.json")
    dataset = _read(dataset_root / "dataset_freeze.json")
    mvp = _read(mvp_root / "mvp_acceptance.json")
    masked_preflight = _read(stage3_root / "preflight_masked_conditional.json")
    direct_preflight = _read(stage3_root / "preflight_direct.json")
    if contract["final_data_exists"]:
        raise ValueError("Stage 3 final data was unexpectedly opened.")
    if dataset["dataset_instance_count"] != 384 or dataset["labeled_instance_count"] != 384:
        raise ValueError("Stage 3 development dataset is incomplete.")
    if not mvp["passed"] or not masked_preflight["passed"] or not direct_preflight["passed"]:
        raise ValueError("A Stage 3 pre-training acceptance gate failed.")

    tracked = [
        "src/gdm_factor_diffusion/common/config.py",
        "src/gdm_factor_diffusion/diffusion/partial_mask.py",
        "src/gdm_factor_diffusion/experiments/schema.py",
        "src/gdm_factor_diffusion/graph/partial_context.py",
        "src/gdm_factor_diffusion/models/conditional_denoiser.py",
        "src/gdm_factor_diffusion/training/masked_objectives.py",
        "src/gdm_factor_diffusion/training/masked_trainer.py",
        "src/gdm_factor_diffusion/training/stage3_production.py",
        "src/gdm_factor_diffusion/inference/masked_decode.py",
        "configs/dataset_phase6e_e_stage3_development.yaml",
        "configs/training_phase6e_e_stage3_pilot.yaml",
        "configs/phase6e_e_stage3_contract.yaml",
        "scripts/62_prepare_phase6e_e_stage3.py",
        "scripts/63_validate_phase6e_e_stage3_mvp.py",
        "scripts/64_train_phase6e_e_stage3.py",
        "scripts/65_finalize_phase6e_e_stage3_pretraining.py",
        "active_stage3/run_training.ps1",
        "tests/test_partial_mask_diffusion.py",
        "tests/test_conditional_denoiser.py",
        "tests/test_masked_decode.py",
        "tests/test_stage3_contract.py",
    ]
    evidence_paths = {
        "stage3_contract_lock": stage3_root / "stage3_contract_lock.json",
        "dataset_freeze": dataset_root / "dataset_freeze.json",
        "mvp_acceptance": mvp_root / "mvp_acceptance.json",
    }
    freeze = {
        "schema_version": "1.0",
        "phase": "6E-E Stage 3 training-ready freeze",
        "regression_test_count": 117,
        "development_instances": 384,
        "verified_solution_count": dataset["verified_solution_count"],
        "target_policy": "best",
        "formal_training_started": False,
        "future_final_data_generated": False,
        "source_sha256": {name: _sha256(root / name) for name in tracked},
        "evidence_sha256": {
            name: _sha256(path) for name, path in evidence_paths.items()
        },
        "parameter_counts": {
            "masked_conditional": masked_preflight["parameter_count"],
            "direct": direct_preflight["parameter_count"],
        },
    }
    destination = stage3_root / "pretraining_freeze.json"
    if destination.exists() and _read(destination) != freeze:
        raise ValueError("Existing pre-training freeze disagrees with current state.")
    write_json(destination, freeze)
    print(
        f"phase={freeze['phase']} tests={freeze['regression_test_count']} "
        f"instances={freeze['development_instances']} training_started=False"
    )
    print(f"freeze={destination} sha256={_sha256(destination)}")


if __name__ == "__main__":
    main()
