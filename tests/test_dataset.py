from pathlib import Path

import pytest

from gdm_factor_diffusion.data import (
    audit_dataset_config_contract,
    generate_dataset,
    load_manifest,
    load_partition,
)


def _dataset_config(output: Path) -> dict:
    common = {
        "share_probability": [0.4, 0.7],
        "compatibility_density": [0.4, 0.7],
        "topology_density": [0.3, 0.6],
        "capacity_slack": [0.2, 0.4],
        "minimum_candidates": 2,
    }
    return {
        "dataset": {
            "name": "unit-dataset",
            "base_seed": 77,
            "output": str(output),
            "partitions": {
                "train": {
                    **common,
                    "role": "train",
                    "regime": "in_distribution",
                    "size_profile": "small",
                    "count": 2,
                    "num_applications": 2,
                    "num_devices": 5,
                    "application_type_ids": [0, 1],
                },
                "validation": {
                    **common,
                    "role": "validation",
                    "regime": "in_distribution",
                    "size_profile": "small",
                    "count": 1,
                    "num_applications": 2,
                    "num_devices": 5,
                    "application_type_ids": [2, 3],
                },
                "test_unseen_size": {
                    **common,
                    "role": "test",
                    "regime": "unseen_size",
                    "size_profile": "large",
                    "count": 1,
                    "num_applications": 4,
                    "num_devices": 8,
                    "service_count_range": [15, 24],
                    "application_type_ids": [0, 1, 2, 5],
                },
            },
        }
    }


def test_dataset_manifest_partitions_and_loading_are_reproducible(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = generate_dataset(_dataset_config(first_root), output_root=first_root)
    second = generate_dataset(_dataset_config(second_root), output_root=second_root)

    assert first == second
    loaded_manifest = load_manifest(first_root / "manifest.json")
    assert loaded_manifest["instance_count"] == 4
    assert len({entry["seed"] for entry in loaded_manifest["instances"]}) == 4
    assert len({entry["instance_id"] for entry in loaded_manifest["instances"]}) == 4
    assert all(not entry["witness_is_model_input"] for entry in first["instances"])
    assert all(entry["graph_readiness"]["ready"] for entry in first["instances"])

    train = load_partition(first_root, "train")
    validation = load_partition(first_root, "validation")
    test = load_partition(first_root, "test_unseen_size")
    assert len(train) == 2
    assert len(validation) == 1
    assert len(test) == 1
    assert all("witness_placement" not in instance.metadata for instance in train)
    assert test[0].num_applications > train[0].num_applications
    assert 15 <= test[0].num_services <= 24
    assert all("generation_attempt" in entry for entry in first["instances"])

    first_entry = loaded_manifest["instances"][0]
    instance_path = first_root / first_entry["path"]
    instance_path.write_bytes(instance_path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_partition(first_root, first_entry["partition"])


def test_final_main_contract_rejects_confounded_shift() -> None:
    config = _dataset_config(Path("unused"))
    dataset = config["dataset"]
    dataset["partitions"]["validation"]["application_type_ids"] = [0, 1]
    dataset["contract"] = {
        "family": "final_main",
        "expected_instance_count": 4,
        "labeling": {
            "target_size": 2,
            "total_time_limit_seconds": 5,
            "threads": 1,
        },
        "base_partition": "train",
        "controlled_shifts": {
            "validation": [],
            "test_unseen_size": ["service_count_range", "num_applications"],
        },
    }
    with pytest.raises(ValueError, match="unapproved field 'num_devices'"):
        audit_dataset_config_contract(config)


def test_final_scale_contract_requires_test_only_partitions() -> None:
    config = _dataset_config(Path("unused"))
    dataset = config["dataset"]
    dataset["contract"] = {
        "family": "final_scale",
        "expected_instance_count": 4,
        "labeling": {
            "target_size": 2,
            "total_time_limit_seconds": 5,
            "threads": 1,
        },
    }
    with pytest.raises(ValueError, match="test-only"):
        audit_dataset_config_contract(config)
