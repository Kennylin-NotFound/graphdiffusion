from copy import deepcopy
from pathlib import Path

import pytest

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.data.contracts import audit_dataset_config_contract
from gdm_factor_diffusion.experiments.phase6ee_stage3 import prepare_stage3_contract


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]


def test_stage3_development_contract_is_balanced_and_isolated() -> None:
    config = load_config(
        IMPLEMENTATION_ROOT
        / "configs"
        / "dataset_phase6e_e_stage3_development.yaml"
    )
    audit_dataset_config_contract(config)
    contract = config["dataset"]["contract"]
    assert contract["expected_instance_count"] == 384
    assert contract["partition_counts"] == {
        "train": 256,
        "checkpoint_selection": 64,
        "pilot_gate": 64,
    }


def test_stage3_contract_rejects_distribution_drift_between_splits() -> None:
    config = load_config(
        IMPLEMENTATION_ROOT
        / "configs"
        / "dataset_phase6e_e_stage3_development.yaml"
    )
    changed = deepcopy(config)
    changed["dataset"]["partitions"]["pilot_gate"]["topology_density"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="changes distribution field"):
        audit_dataset_config_contract(changed)


def test_stage3_lock_is_idempotent_and_keeps_final_data_closed() -> None:
    first = prepare_stage3_contract(IMPLEMENTATION_ROOT)
    second = prepare_stage3_contract(IMPLEMENTATION_ROOT)
    assert first == second
    assert not first["final_data_exists"]
    assert first["prior_frozen_sha256"]["stage2b_lock.json"].startswith("F3610439")
