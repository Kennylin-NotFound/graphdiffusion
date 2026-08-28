"""Framework-independent typed factor-graph indices and static edge attributes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gdm_factor_diffusion.solver.pair_costs import build_dependency_pair_costs

from .schema import DeploymentInstance


@dataclass(frozen=True, slots=True)
class FactorGraphBlueprint:
    node_counts: dict[str, int]
    relation_index: dict[str, np.ndarray]
    candidate_processing_latency: np.ndarray
    membership_sink_indicator: np.ndarray
    physical_link_rate: np.ndarray
    dependency_pair_admissible: np.ndarray
    dependency_pair_latency: np.ndarray


def _edge_index(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.vstack((source, target)), dtype=np.int64)


def build_factor_graph_blueprint(instance: DeploymentInstance) -> FactorGraphBlueprint:
    """Build static typed relations without committing to a graph framework."""

    service, device = np.nonzero(instance.compatibility_mask)
    dependency = np.arange(instance.num_dependencies, dtype=np.int64)
    upstream = instance.dependency_index[0]
    downstream = instance.dependency_index[1]
    application, member_service = np.nonzero(instance.membership)
    link_source, link_target = np.nonzero(instance.connectivity)

    relation_index = {
        "service__compatible_with__device": _edge_index(service, device),
        "device__candidate_for__service": _edge_index(device, service),
        "service__upstream_of__dependency": _edge_index(upstream, dependency),
        "service__downstream_of__dependency": _edge_index(downstream, dependency),
        "dependency__to_upstream__service": _edge_index(dependency, upstream),
        "dependency__to_downstream__service": _edge_index(dependency, downstream),
        "service__member_of__application": _edge_index(member_service, application),
        "application__contains__service": _edge_index(application, member_service),
        "device__linked_to__device": _edge_index(link_source, link_target),
    }
    pair_costs = build_dependency_pair_costs(instance)
    return FactorGraphBlueprint(
        node_counts={
            "service": instance.num_services,
            "device": instance.num_devices,
            "dependency": instance.num_dependencies,
            "application": instance.num_applications,
        },
        relation_index=relation_index,
        candidate_processing_latency=instance.processing_latency[service, device].astype(
            np.float32
        ),
        membership_sink_indicator=instance.sink_mask[
            application, member_service
        ].astype(np.float32),
        physical_link_rate=instance.link_rate[link_source, link_target].astype(
            np.float32
        ),
        dependency_pair_admissible=pair_costs.admissible,
        dependency_pair_latency=pair_costs.transmission_latency,
    )
