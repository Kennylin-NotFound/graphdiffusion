"""Audit whether an instance exposes every input required by the future factor graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gdm_factor_diffusion.solver.placement_verifier import verify_placement

from .graph_blueprint import build_factor_graph_blueprint
from .schema import DeploymentInstance


@dataclass(frozen=True, slots=True)
class GraphReadinessReport:
    ready: bool
    errors: tuple[str, ...]
    dimensions: dict[str, int]
    relation_counts: dict[str, int]
    statistics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "errors": list(self.errors),
            "dimensions": self.dimensions,
            "relation_counts": self.relation_counts,
            "statistics": self.statistics,
        }


def audit_graph_readiness(
    instance: DeploymentInstance,
    witness_placement: np.ndarray | None = None,
) -> GraphReadinessReport:
    """Check the explicit arrays needed by the typed assignment factor graph."""

    errors: list[str] = []
    try:
        instance.validate()
    except ValueError as error:
        errors.append(f"schema: {error}")

    for key in ("service_feature_names", "device_feature_names", "resource_names"):
        if key not in instance.metadata:
            errors.append(f"metadata missing {key}")

    blueprint = build_factor_graph_blueprint(instance)
    if not np.isfinite(blueprint.dependency_pair_latency).all():
        errors.append("dependency pair costs contain non-finite values")
    if not instance.compatibility_mask.any(axis=1).all():
        errors.append("at least one service has no categorical choices")
    if not instance.membership.any(axis=0).all():
        errors.append("at least one service has no application-factor relation")

    if witness_placement is not None:
        verification = verify_placement(instance, witness_placement)
        if not verification.feasible:
            errors.append("generation witness does not pass the shared verifier")

    candidate_edges = int(
        blueprint.relation_index["service__compatible_with__device"].shape[1]
    )
    physical_links = int(
        blueprint.relation_index["device__linked_to__device"].shape[1]
    )
    application_memberships = int(
        blueprint.relation_index["service__member_of__application"].shape[1]
    )
    sink_memberships = int(instance.sink_mask.sum())
    dimensions = {
        "services": instance.num_services,
        "devices": instance.num_devices,
        "dependencies": instance.num_dependencies,
        "applications": instance.num_applications,
        "resources": instance.num_resources,
        "service_features": int(instance.service_features.shape[1]),
        "device_features": int(instance.device_features.shape[1]),
    }
    relation_counts = {
        "service_to_device_candidates": candidate_edges,
        "device_to_service_candidates": candidate_edges,
        "service_to_dependency_endpoints": 2 * instance.num_dependencies,
        "dependency_to_service_endpoints": 2 * instance.num_dependencies,
        "service_to_application_memberships": application_memberships,
        "application_to_service_memberships": application_memberships,
        "sink_memberships": sink_memberships,
        "directed_device_links": physical_links,
    }
    possible_physical_links = instance.num_devices * max(instance.num_devices - 1, 1)
    statistics = {
        "candidate_choices_per_service": candidate_edges / instance.num_services,
        "compatibility_density": float(instance.compatibility_mask.mean()),
        "directed_topology_density": physical_links / possible_physical_links,
        "shared_service_fraction": float((instance.membership.sum(axis=0) > 1).mean()),
        "admissible_pair_fraction": float(blueprint.dependency_pair_admissible.mean())
        if blueprint.dependency_pair_admissible.size
        else 0.0,
    }
    return GraphReadinessReport(
        ready=not errors,
        errors=tuple(errors),
        dimensions=dimensions,
        relation_counts=relation_counts,
        statistics=statistics,
    )
