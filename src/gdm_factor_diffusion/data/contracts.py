"""Auditable contracts for final scientific dataset configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DISTRIBUTION_KEYS = (
    "service_count_range",
    "num_applications",
    "num_devices",
    "share_probability",
    "compatibility_density",
    "topology_density",
    "capacity_slack",
    "minimum_candidates",
    "application_type_pool",
    "application_type_ids",
)


def audit_dataset_config_contract(
    config: Mapping[str, Any],
    *,
    observed_instance_count: int | None = None,
) -> None:
    """Validate optional final-data count, role, and controlled-shift contracts."""

    dataset = config["dataset"]
    contract = dataset.get("contract")
    if contract is None:
        return
    partitions = dataset["partitions"]
    expected_count = int(contract["expected_instance_count"])
    configured_count = sum(int(partition["count"]) for partition in partitions.values())
    if expected_count != configured_count:
        raise ValueError(
            "Dataset contract expected_instance_count disagrees with partition counts."
        )
    if observed_instance_count is not None and expected_count != observed_instance_count:
        raise ValueError(
            "Dataset contract expected_instance_count disagrees with the manifest."
        )
    labeling = contract["labeling"]
    if int(labeling["target_size"]) < 1:
        raise ValueError("Dataset contract labeling target_size must be positive.")
    if float(labeling["total_time_limit_seconds"]) <= 0:
        raise ValueError("Dataset contract labeling time limit must be positive.")
    if int(labeling["threads"]) < 1:
        raise ValueError("Dataset contract labeling threads must be positive.")

    family = str(contract["family"])
    if family == "final_main":
        _audit_final_main_contract(partitions, contract)
    elif family == "final_scale":
        if any(partition["role"] != "test" for partition in partitions.values()):
            raise ValueError("Every final-scale partition must be test-only.")
    elif family == "stage3_development":
        _audit_stage3_development_contract(partitions, contract)
    elif family == "stage36_efficiency_confirmation":
        _audit_stage36_confirmation_contract(partitions, contract)
    elif family == "stage38_sealed_confirmation":
        _audit_stage38_sealed_contract(partitions, contract)
    elif family == "realistic_profile":
        _audit_realistic_profile_contract(partitions, contract)
    else:
        raise ValueError(f"Unknown dataset contract family: {family!r}.")


def _audit_stage36_confirmation_contract(
    partitions: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    expected_name = "efficiency_confirmation"
    if set(partitions) != {expected_name}:
        raise ValueError(
            "Stage 3.6 confirmation data must contain only efficiency_confirmation."
        )
    expected_counts = {
        str(name): int(value)
        for name, value in contract["partition_counts"].items()
    }
    if expected_counts != {expected_name: int(partitions[expected_name]["count"])}:
        raise ValueError("Stage 3.6 confirmation partition count is inconsistent.")
    partition = partitions[expected_name]
    if partition["role"] != "test":
        raise ValueError("Stage 3.6 confirmation partition must be test-only.")
    if partition["regime"] != "in_distribution":
        raise ValueError("Stage 3.6 confirmation data must remain in-distribution.")


def _audit_stage38_sealed_contract(
    partitions: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    expected_name = "sealed_test_id"
    if set(partitions) != {expected_name}:
        raise ValueError(
            "Stage 3.8 sealed data must contain only sealed_test_id."
        )
    expected_counts = {
        str(name): int(value)
        for name, value in contract["partition_counts"].items()
    }
    if expected_counts != {expected_name: int(partitions[expected_name]["count"])}:
        raise ValueError("Stage 3.8 sealed partition count is inconsistent.")
    partition = partitions[expected_name]
    if partition["role"] != "test":
        raise ValueError("Stage 3.8 sealed partition must be test-only.")
    if partition["regime"] != "in_distribution":
        raise ValueError("Stage 3.8 sealed data must remain in-distribution.")


def _audit_stage3_development_contract(
    partitions: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    expected_names = {"train", "checkpoint_selection", "pilot_gate"}
    if set(partitions) != expected_names:
        raise ValueError(
            "Stage 3 development partitions must be train, checkpoint_selection, "
            "and pilot_gate."
        )
    expected_roles = {
        "train": "train",
        "checkpoint_selection": "validation",
        "pilot_gate": "test",
    }
    expected_counts = {
        str(name): int(value)
        for name, value in contract["partition_counts"].items()
    }
    if expected_counts.keys() != expected_names:
        raise ValueError("Stage 3 partition_counts must enumerate every partition.")
    base = partitions["train"]
    for name in sorted(expected_names):
        partition = partitions[name]
        if partition["role"] != expected_roles[name]:
            raise ValueError(f"Stage 3 partition {name!r} has the wrong role.")
        if int(partition["count"]) != expected_counts[name]:
            raise ValueError(f"Stage 3 partition {name!r} has the wrong count.")
        if partition["regime"] != "in_distribution":
            raise ValueError("Stage 3 development data must remain in-distribution.")
        for key in DISTRIBUTION_KEYS:
            if partition.get(key) != base.get(key):
                raise ValueError(
                    f"Stage 3 partition {name!r} changes distribution field {key!r}."
                )


def _audit_final_main_contract(
    partitions: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    base_name = str(contract["base_partition"])
    if base_name not in partitions:
        raise ValueError("Final-main base partition is missing.")
    base = partitions[base_name]
    if base["role"] != "train":
        raise ValueError("Final-main base partition must have the train role.")

    shifts = contract["controlled_shifts"]
    nonbase = set(partitions) - {base_name}
    if set(shifts) != nonbase:
        raise ValueError(
            "Final-main controlled_shifts must enumerate every non-training partition."
        )
    known_keys = set(DISTRIBUTION_KEYS)
    for name, allowed_changes in shifts.items():
        allowed = set(allowed_changes)
        unknown = allowed - known_keys
        if unknown:
            raise ValueError(
                f"Controlled shift {name!r} uses unknown distribution keys: "
                f"{sorted(unknown)}"
            )
        partition = partitions[name]
        for key in DISTRIBUTION_KEYS:
            if key not in allowed and partition.get(key) != base.get(key):
                raise ValueError(
                    f"Controlled shift {name!r} changes unapproved field {key!r}."
                )


def _audit_realistic_profile_contract(
    partitions: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    if any(partition["role"] != "test" for partition in partitions.values()):
        raise ValueError("Every realistic-profile partition must be test-only.")
    profiles = contract.get("realistic_profiles")
    if profiles is None:
        raise ValueError("Realistic-profile contract must define realistic_profiles.")
    if set(profiles) != set(partitions):
        raise ValueError(
            "Realistic-profile contract must enumerate every configured partition."
        )
