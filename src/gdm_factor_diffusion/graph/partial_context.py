"""Dynamic graph context derived from an explicit partial placement."""

from __future__ import annotations

import torch
from torch import Tensor

from gdm_factor_diffusion.diffusion.partial_mask import (
    PartialPlacementState,
    validate_partial_state,
)

from .batch_adapter import FactorGraphBatch


def build_partial_context(
    batch: FactorGraphBatch,
    state: PartialPlacementState,
) -> dict[str, Tensor]:
    """Build visible placement, residual capacity, and dependency context."""

    validate_partial_state(state, batch.candidate_mask, batch.service_mask)
    dtype = batch.processing_latency.dtype
    committed = state.committed_mask
    assignment = torch.nn.functional.one_hot(
        state.assignment.clamp_min(0),
        num_classes=batch.candidate_mask.shape[-1],
    ).to(dtype=dtype)
    assignment = assignment * committed.unsqueeze(-1).to(dtype=dtype)

    selected_processing = batch.processing_latency.gather(
        -1, state.assignment.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    selected_processing = selected_processing * committed.to(dtype=dtype)
    service_dense = torch.stack(
        (committed.to(dtype=dtype), selected_processing),
        dim=-1,
    )

    resource_load = torch.einsum("bmd,bmr->bdr", assignment, batch.service_demand)
    normalized_load = resource_load / batch.device_capacity.clamp_min(1e-8)
    residual_capacity = (batch.device_capacity - resource_load) / batch.device_capacity.clamp_min(
        1e-8
    )
    device_mask = batch.device_node_index >= 0
    device_dense = torch.cat((normalized_load, residual_capacity), dim=-1)
    device_dense = device_dense * device_mask.unsqueeze(-1).to(dtype=dtype)

    source_service = batch.dependency_index[:, 0].clamp_min(0)
    target_service = batch.dependency_index[:, 1].clamp_min(0)
    source_visible = committed.gather(1, source_service) & batch.dependency_mask
    target_visible = committed.gather(1, target_service) & batch.dependency_mask
    both_visible = source_visible & target_visible
    source_device = state.assignment.gather(1, source_service).clamp_min(0)
    target_device = state.assignment.gather(1, target_service).clamp_min(0)
    batch_index = torch.arange(batch.batch_size, device=state.assignment.device)[:, None]
    edge_index = torch.arange(
        batch.dependency_mask.shape[1], device=state.assignment.device
    )[None, :]
    selected_latency = batch.pair_latency[
        batch_index, edge_index, source_device, target_device
    ]
    selected_admissible = batch.pair_admissible[
        batch_index, edge_index, source_device, target_device
    ]
    selected_latency = selected_latency * both_visible.to(dtype=dtype)
    selected_admissible = selected_admissible & both_visible
    dependency_dense = torch.stack(
        (
            source_visible.to(dtype=dtype),
            target_visible.to(dtype=dtype),
            selected_latency,
            selected_admissible.to(dtype=dtype),
        ),
        dim=-1,
    )
    dependency_dense = dependency_dense * batch.dependency_mask.unsqueeze(-1).to(
        dtype=dtype
    )

    service_context = torch.zeros(
        (batch.graph["service"].num_nodes, 2),
        dtype=dtype,
        device=state.assignment.device,
    )
    device_context = torch.zeros(
        (batch.graph["device"].num_nodes, 2 * batch.service_demand.shape[-1]),
        dtype=dtype,
        device=state.assignment.device,
    )
    dependency_context = torch.zeros(
        (batch.graph["dependency"].num_nodes, 4),
        dtype=dtype,
        device=state.assignment.device,
    )
    service_context[batch.service_node_index[batch.service_mask]] = service_dense[
        batch.service_mask
    ]
    device_context[batch.device_node_index[device_mask]] = device_dense[device_mask]
    dependency_context[batch.dependency_node_index[batch.dependency_mask]] = (
        dependency_dense[batch.dependency_mask]
    )
    return {
        "service": service_context,
        "device": device_context,
        "dependency": dependency_context,
        "service_dense": service_dense,
        "device_dense": device_dense,
        "dependency_dense": dependency_dense,
        "resource_load": resource_load,
    }
