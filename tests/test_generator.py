import numpy as np

from gdm_factor_diffusion.data import (
    InstanceGenerationSpec,
    audit_graph_readiness,
    generate_instance,
)
from gdm_factor_diffusion.solver import evaluate_latency, verify_placement


def _spec(**updates: object) -> InstanceGenerationSpec:
    values: dict[str, object] = {
        "instance_id": "generated-test",
        "seed": 123,
        "partition": "train",
        "role": "train",
        "regime": "in_distribution",
        "size_profile": "seen_small_medium",
        "num_applications": 3,
        "num_devices": 6,
        "share_probability": 0.6,
        "compatibility_density": 0.5,
        "topology_density": 0.4,
        "capacity_slack": 0.3,
        "minimum_candidates": 2,
        "application_type_ids": (0, 1, 5),
    }
    values.update(updates)
    return InstanceGenerationSpec(**values)


def test_generation_is_reproducible_and_witness_is_feasible() -> None:
    first = generate_instance(_spec())
    second = generate_instance(_spec())

    assert first.instance.equivalent_to(second.instance)
    assert np.array_equal(first.witness_placement, second.witness_placement)
    assert verify_placement(first.instance, first.witness_placement).feasible
    assert np.isfinite(evaluate_latency(first.instance, first.witness_placement).objective)
    assert audit_graph_readiness(first.instance, first.witness_placement).ready


def test_safe_sharing_reduces_joint_dag_size() -> None:
    low = generate_instance(
        _spec(
            instance_id="low-share",
            num_applications=2,
            application_type_ids=(0, 1),
            share_probability=0.0,
        )
    )
    high = generate_instance(
        _spec(
            instance_id="high-share",
            num_applications=2,
            application_type_ids=(0, 1),
            share_probability=1.0,
        )
    )

    assert high.instance.num_services < low.instance.num_services
    assert high.summary["sharing_ratio"] > low.summary["sharing_ratio"]
    assert (high.instance.membership.sum(axis=0) > 1).any()


def test_generated_instance_preserves_graph_and_split_metadata() -> None:
    generated = generate_instance(_spec())
    instance = generated.instance

    assert instance.metadata["partition"] == "train"
    assert instance.metadata["role"] == "train"
    assert instance.metadata["regime"] == "in_distribution"
    assert instance.metadata["catalog_version"] == "1.0"
    assert instance.service_type_id.shape == (instance.num_services,)
    assert instance.device_type_id.shape == (instance.num_devices,)
    assert instance.application_type_id.shape == (instance.num_applications,)
    assert "witness_placement" not in instance.metadata


def test_application_type_pool_is_respected() -> None:
    generated = generate_instance(
        _spec(
            instance_id="held-out-workflow",
            num_applications=3,
            application_type_ids=None,
            application_type_pool=(5,),
        )
    )
    assert generated.instance.application_type_id.tolist() == [5, 5, 5]
