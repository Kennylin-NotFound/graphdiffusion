"""Padded categorical tensors aligned with a flattened PyG factor-graph batch."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch

from gdm_factor_diffusion.data.graph_blueprint import build_factor_graph_blueprint
from gdm_factor_diffusion.data.schema import DeploymentInstance
from gdm_factor_diffusion.diffusion.masking import state_to_one_hot, validate_state

from .hetero_graph import build_hetero_graph


@dataclass(slots=True)
class FactorGraphBatch:
    """All static tensors needed by the denoiser for a batch of instances."""

    graph: Any
    candidate_mask: Tensor
    service_mask: Tensor
    service_node_index: Tensor
    device_node_index: Tensor
    dependency_node_index: Tensor
    processing_latency: Tensor
    service_demand: Tensor
    device_capacity: Tensor
    dependency_index: Tensor
    dependency_mask: Tensor
    pair_admissible: Tensor
    pair_latency: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.candidate_mask.shape[0])

    def to(self, device: torch.device | str) -> "FactorGraphBatch":
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device)
        return FactorGraphBatch(**values)

    def selected_device_node_index(self, state: Tensor) -> Tensor:
        """Map local categorical devices to flattened device-node indices."""

        validate_state(state, self.candidate_mask, self.service_mask)
        selected = self.device_node_index.gather(1, state.clamp_min(0))
        return selected.masked_fill(~self.service_mask, -1)


@dataclass(frozen=True, slots=True)
class GraphFeatureSchema:
    """Stable aligned feature columns shared by every training batch."""

    service_feature_names: tuple[str, ...]
    device_feature_names: tuple[str, ...]
    resource_names: tuple[str, ...]


def infer_feature_schema(
    instances: Sequence[DeploymentInstance],
) -> GraphFeatureSchema:
    """Infer one deterministic feature space for a collection of instances."""

    if not instances:
        raise ValueError("At least one instance is required.")
    resource_names = tuple(instances[0].metadata["resource_names"])
    if any(tuple(instance.metadata["resource_names"]) != resource_names for instance in instances):
        raise ValueError("All instances must use the same ordered resource names.")
    service_feature_names = tuple(
        sorted(
            {
                name
                for instance in instances
                for name in (
                    list(instance.metadata["service_feature_names"])
                    + [f"demand:{resource}" for resource in resource_names]
                )
            }
        )
    )
    device_feature_names = tuple(
        sorted(
            {
                name
                for instance in instances
                for name in (
                    list(instance.metadata["device_feature_names"])
                    + [f"capacity:{resource}" for resource in resource_names]
                )
            }
        )
    )
    return GraphFeatureSchema(
        service_feature_names=service_feature_names,
        device_feature_names=device_feature_names,
        resource_names=resource_names,
    )


def merge_feature_schemas(
    schemas: Sequence[GraphFeatureSchema],
) -> GraphFeatureSchema:
    """Merge partition schemas while preserving one ordered resource contract."""

    if not schemas:
        raise ValueError("At least one feature schema is required.")
    resource_names = schemas[0].resource_names
    if any(schema.resource_names != resource_names for schema in schemas):
        raise ValueError("Feature schemas use different resource names.")
    return GraphFeatureSchema(
        service_feature_names=tuple(
            sorted(
                {
                    name
                    for schema in schemas
                    for name in schema.service_feature_names
                }
            )
        ),
        device_feature_names=tuple(
            sorted(
                {
                    name
                    for schema in schemas
                    for name in schema.device_feature_names
                }
            )
        ),
        resource_names=resource_names,
    )


def _node_index_table(
    counts: list[int],
    maximum: int,
) -> Tensor:
    table = torch.full((len(counts), maximum), -1, dtype=torch.long)
    offset = 0
    for batch_index, count in enumerate(counts):
        table[batch_index, :count] = torch.arange(count, dtype=torch.long) + offset
        offset += count
    return table


def build_factor_graph_batch(
    instances: Sequence[DeploymentInstance],
    *,
    feature_schema: GraphFeatureSchema | None = None,
) -> FactorGraphBatch:
    """Batch immutable instances into PyG relations and aligned padded tensors."""

    if not instances:
        raise ValueError("At least one instance is required.")
    feature_schema = feature_schema or infer_feature_schema(instances)
    graphs = [
        build_hetero_graph(
            instance,
            service_feature_names=feature_schema.service_feature_names,
            device_feature_names=feature_schema.device_feature_names,
        )
        for instance in instances
    ]
    graph_batch = Batch.from_data_list(graphs)
    batch_size = len(instances)
    max_services = max(instance.num_services for instance in instances)
    max_devices = max(instance.num_devices for instance in instances)
    max_dependencies = max(instance.num_dependencies for instance in instances)
    resources = len(feature_schema.resource_names)
    for instance in instances:
        if tuple(instance.metadata["resource_names"]) != feature_schema.resource_names:
            raise ValueError("Instance resource names disagree with the feature schema.")

    candidate_mask = torch.zeros(
        (batch_size, max_services, max_devices), dtype=torch.bool
    )
    processing_latency = torch.zeros(
        (batch_size, max_services, max_devices), dtype=torch.float32
    )
    service_demand = torch.zeros(
        (batch_size, max_services, resources), dtype=torch.float32
    )
    device_capacity = torch.zeros(
        (batch_size, max_devices, resources), dtype=torch.float32
    )
    dependency_index = torch.full(
        (batch_size, 2, max_dependencies), -1, dtype=torch.long
    )
    dependency_mask = torch.zeros(
        (batch_size, max_dependencies), dtype=torch.bool
    )
    pair_admissible = torch.zeros(
        (batch_size, max_dependencies, max_devices, max_devices),
        dtype=torch.bool,
    )
    pair_latency = torch.zeros_like(pair_admissible, dtype=torch.float32)

    for batch_index, instance in enumerate(instances):
        m, d, e = (
            instance.num_services,
            instance.num_devices,
            instance.num_dependencies,
        )
        candidate_mask[batch_index, :m, :d] = torch.from_numpy(
            instance.compatibility_mask
        )
        processing_latency[batch_index, :m, :d] = torch.from_numpy(
            instance.processing_latency
        )
        service_demand[batch_index, :m] = torch.from_numpy(instance.service_demand)
        device_capacity[batch_index, :d] = torch.from_numpy(instance.device_capacity)
        if e:
            dependency_index[batch_index, :, :e] = torch.from_numpy(
                instance.dependency_index
            )
            dependency_mask[batch_index, :e] = True
            blueprint = build_factor_graph_blueprint(instance)
            pair_admissible[batch_index, :e, :d, :d] = torch.from_numpy(
                blueprint.dependency_pair_admissible
            )
            pair_latency[batch_index, :e, :d, :d] = torch.from_numpy(
                blueprint.dependency_pair_latency
            ).float()

    service_mask = candidate_mask.any(dim=-1)
    return FactorGraphBatch(
        graph=graph_batch,
        candidate_mask=candidate_mask,
        service_mask=service_mask,
        service_node_index=_node_index_table(
            [instance.num_services for instance in instances], max_services
        ),
        device_node_index=_node_index_table(
            [instance.num_devices for instance in instances], max_devices
        ),
        dependency_node_index=_node_index_table(
            [instance.num_dependencies for instance in instances], max_dependencies
        ),
        processing_latency=processing_latency,
        service_demand=service_demand,
        device_capacity=device_capacity,
        dependency_index=dependency_index,
        dependency_mask=dependency_mask,
        pair_admissible=pair_admissible,
        pair_latency=pair_latency,
    )


def build_dynamic_context(
    batch: FactorGraphBatch,
    noisy_state: Tensor,
) -> dict[str, Tensor]:
    """Build current processing, resource-load, and dependency-pair context."""

    validate_state(noisy_state, batch.candidate_mask, batch.service_mask)
    dtype = batch.processing_latency.dtype
    assignment = state_to_one_hot(
        noisy_state,
        batch.candidate_mask,
        batch.service_mask,
        dtype=dtype,
    )
    selected_processing = batch.processing_latency.gather(
        -1, noisy_state.clamp_min(0).unsqueeze(-1)
    )
    selected_processing = selected_processing * batch.service_mask.unsqueeze(-1)

    resource_load = torch.einsum("bmd,bmr->bdr", assignment, batch.service_demand)
    normalized_load = resource_load / batch.device_capacity.clamp_min(1e-8)
    device_mask = batch.device_node_index >= 0
    normalized_load = normalized_load * device_mask.unsqueeze(-1)

    source_service = batch.dependency_index[:, 0].clamp_min(0)
    target_service = batch.dependency_index[:, 1].clamp_min(0)
    source_device = noisy_state.gather(1, source_service).clamp_min(0)
    target_device = noisy_state.gather(1, target_service).clamp_min(0)
    batch_index = torch.arange(batch.batch_size, device=noisy_state.device)[:, None]
    edge_index = torch.arange(
        batch.dependency_mask.shape[1], device=noisy_state.device
    )[None, :]
    selected_admissible = batch.pair_admissible[
        batch_index, edge_index, source_device, target_device
    ]
    selected_latency = batch.pair_latency[
        batch_index, edge_index, source_device, target_device
    ]
    dependency_context = torch.stack(
        (selected_latency, selected_admissible.to(dtype=dtype)), dim=-1
    )
    dependency_context = dependency_context * batch.dependency_mask.unsqueeze(-1)

    total_service = batch.graph["service"].num_nodes
    total_device = batch.graph["device"].num_nodes
    total_dependency = batch.graph["dependency"].num_nodes
    service_context = torch.zeros(
        (total_service, 1), dtype=dtype, device=noisy_state.device
    )
    device_context = torch.zeros(
        (total_device, batch.service_demand.shape[-1]),
        dtype=dtype,
        device=noisy_state.device,
    )
    dependency_flat = torch.zeros(
        (total_dependency, 2), dtype=dtype, device=noisy_state.device
    )
    service_context[batch.service_node_index[batch.service_mask]] = selected_processing[
        batch.service_mask
    ]
    device_context[batch.device_node_index[device_mask]] = normalized_load[device_mask]
    dependency_flat[batch.dependency_node_index[batch.dependency_mask]] = (
        dependency_context[batch.dependency_mask]
    )
    return {
        "service": service_context,
        "device": device_context,
        "dependency": dependency_flat,
    }
