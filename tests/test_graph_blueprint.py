import numpy as np

from gdm_factor_diffusion.data import (
    InstanceGenerationSpec,
    build_factor_graph_blueprint,
    generate_instance,
)


def test_factor_graph_blueprint_relations_and_attributes_align() -> None:
    generated = generate_instance(
        InstanceGenerationSpec(
            instance_id="blueprint-test",
            seed=9,
            partition="train",
            role="train",
            regime="in_distribution",
            size_profile="small",
            num_applications=2,
            num_devices=5,
            share_probability=0.8,
            compatibility_density=0.5,
            topology_density=0.4,
            capacity_slack=0.3,
            application_type_ids=(0, 1),
        )
    )
    instance = generated.instance
    blueprint = build_factor_graph_blueprint(instance)

    candidate = blueprint.relation_index["service__compatible_with__device"]
    member = blueprint.relation_index["service__member_of__application"]
    linked = blueprint.relation_index["device__linked_to__device"]

    assert blueprint.node_counts == {
        "service": instance.num_services,
        "device": instance.num_devices,
        "dependency": instance.num_dependencies,
        "application": instance.num_applications,
    }
    assert candidate.shape[1] == int(instance.compatibility_mask.sum())
    assert len(blueprint.candidate_processing_latency) == candidate.shape[1]
    assert np.all(blueprint.candidate_processing_latency > 0)
    assert member.shape[1] == int(instance.membership.sum())
    assert len(blueprint.membership_sink_indicator) == member.shape[1]
    assert linked.shape[1] == int(instance.connectivity.sum())
    assert len(blueprint.physical_link_rate) == linked.shape[1]
    assert blueprint.dependency_pair_latency.shape == (
        instance.num_dependencies,
        instance.num_devices,
        instance.num_devices,
    )
    assert np.isfinite(blueprint.dependency_pair_latency).all()
