"""Convert the framework-independent blueprint to a PyG heterogeneous graph."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import HeteroData

from gdm_factor_diffusion.data.graph_blueprint import build_factor_graph_blueprint
from gdm_factor_diffusion.data.schema import DeploymentInstance


def _relation_tuple(name: str) -> tuple[str, str, str]:
    source, relation, target = name.split("__")
    return source, relation, target


def _align_features(
    values: np.ndarray,
    source_names: tuple[str, ...],
    target_names: tuple[str, ...] | None,
) -> np.ndarray:
    if target_names is None:
        return values
    source_index = {name: index for index, name in enumerate(source_names)}
    missing = set(source_names) - set(target_names)
    if missing:
        raise ValueError(f"Target feature space is missing columns: {sorted(missing)}")
    aligned = np.zeros((values.shape[0], len(target_names)), dtype=np.float32)
    for target_index, name in enumerate(target_names):
        if name in source_index:
            aligned[:, target_index] = values[:, source_index[name]]
    return aligned


def build_hetero_graph(
    instance: DeploymentInstance,
    *,
    service_feature_names: tuple[str, ...] | None = None,
    device_feature_names: tuple[str, ...] | None = None,
) -> HeteroData:
    """Build static node features, typed relations, and scalar edge attributes."""

    blueprint = build_factor_graph_blueprint(instance)
    graph = HeteroData()
    resource_names = tuple(instance.metadata["resource_names"])
    local_service_names = tuple(instance.metadata["service_feature_names"]) + tuple(
        f"demand:{name}" for name in resource_names
    )
    local_device_names = tuple(instance.metadata["device_feature_names"]) + tuple(
        f"capacity:{name}" for name in resource_names
    )
    service_values = np.concatenate(
        (instance.service_features, instance.service_demand), axis=1
    )
    device_values = np.concatenate(
        (instance.device_features, instance.device_capacity), axis=1
    )
    graph["service"].x = torch.from_numpy(
        _align_features(
            service_values,
            local_service_names,
            service_feature_names,
        )
    ).float()
    graph["service"].type_id = torch.from_numpy(instance.service_type_id).long()
    graph["device"].x = torch.from_numpy(
        _align_features(
            device_values,
            local_device_names,
            device_feature_names,
        )
    ).float()
    graph["device"].type_id = torch.from_numpy(instance.device_type_id).long()
    graph["dependency"].x = torch.from_numpy(
        instance.dependency_data_volume[:, None]
    ).float()
    graph["dependency"].type_id = torch.zeros(
        instance.num_dependencies, dtype=torch.long
    )
    graph["application"].x = torch.from_numpy(
        instance.application_weight[:, None]
    ).float()
    graph["application"].type_id = torch.from_numpy(
        instance.application_type_id
    ).long()

    for name, index in blueprint.relation_index.items():
        relation = _relation_tuple(name)
        graph[relation].edge_index = torch.from_numpy(index).long()
        edge_count = index.shape[1]
        if "compatible_with" in name or "candidate_for" in name:
            edge_attr = blueprint.candidate_processing_latency
        elif "member_of" in name or "contains" in name:
            edge_attr = blueprint.membership_sink_indicator
        elif "linked_to" in name:
            edge_attr = blueprint.physical_link_rate
        else:
            edge_attr = np.zeros(edge_count, dtype=np.float32)
        graph[relation].edge_attr = torch.from_numpy(edge_attr[:, None]).float()

    graph.instance_id = instance.instance_id
    graph.service_feature_names = service_feature_names or local_service_names
    graph.device_feature_names = device_feature_names or local_device_names
    return graph
