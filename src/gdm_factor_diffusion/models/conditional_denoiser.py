"""Typed factor-graph predictor conditioned on a partial placement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from gdm_factor_diffusion.data.catalogs import DEVICE_TYPES, SERVICE_TYPES
from gdm_factor_diffusion.diffusion.partial_mask import (
    PartialPlacementState,
    validate_partial_state,
)
from gdm_factor_diffusion.graph.batch_adapter import FactorGraphBatch
from gdm_factor_diffusion.graph.partial_context import build_partial_context

from .factor_layer import TypedFactorLayer


@dataclass(frozen=True, slots=True)
class ConditionalDenoiserConfig:
    num_mask_steps: int = 8
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        if self.num_mask_steps < 1:
            raise ValueError("num_mask_steps must be positive.")
        if self.hidden_dim < 1 or self.num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")


class _ConditionalNodeEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        type_count: int | None,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.feature = nn.Linear(feature_dim, hidden_dim)
        self.type_embedding = (
            nn.Embedding(type_count, hidden_dim) if type_count is not None else None
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, feature: Tensor, type_id: Tensor) -> Tensor:
        encoded = self.feature(feature)
        if self.type_embedding is not None:
            encoded = encoded + self.type_embedding(type_id)
        return torch.nn.functional.gelu(self.normalization(encoded))


class TypedFactorConditionalDenoiser(nn.Module):
    """Predict hidden service placements from the currently committed subset."""

    def __init__(
        self,
        *,
        node_feature_dims: dict[str, int],
        node_type_counts: dict[str, int | None],
        edge_types: tuple[tuple[str, str, str], ...],
        resource_dim: int,
        config: ConditionalDenoiserConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ConditionalDenoiserConfig()
        self.config.validate()
        hidden_dim = self.config.hidden_dim
        self.node_types = tuple(node_feature_dims)
        self.edge_types = edge_types
        self.node_encoder = nn.ModuleDict(
            {
                node_type: _ConditionalNodeEncoder(
                    node_feature_dims[node_type],
                    node_type_counts[node_type],
                    hidden_dim,
                )
                for node_type in self.node_types
            }
        )
        self.timestep_embedding = nn.Embedding(
            self.config.num_mask_steps + 1, hidden_dim
        )
        self.service_dynamic = nn.Linear(2, hidden_dim)
        self.device_dynamic = nn.Linear(2 * resource_dim, hidden_dim)
        self.dependency_dynamic = nn.Linear(4, hidden_dim)
        self.selected_device = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mask_embedding = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_embedding, std=0.02)
        self.layers = nn.ModuleList(
            TypedFactorLayer(
                self.node_types,
                self.edge_types,
                hidden_dim,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_layers)
        )
        self.choice_service = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.choice_device = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.choice_processing = nn.Linear(1, hidden_dim, bias=False)
        self.choice_output = nn.Linear(hidden_dim, 1)

    @classmethod
    def from_batch(
        cls,
        batch: FactorGraphBatch,
        config: ConditionalDenoiserConfig | None = None,
    ) -> "TypedFactorConditionalDenoiser":
        graph = batch.graph
        return cls(
            node_feature_dims={
                node_type: int(graph[node_type].x.shape[-1])
                for node_type in graph.node_types
            },
            node_type_counts={
                "service": len(SERVICE_TYPES),
                "device": len(DEVICE_TYPES),
                "dependency": None,
                "application": None,
            },
            edge_types=tuple(graph.edge_types),
            resource_dim=int(batch.service_demand.shape[-1]),
            config=config,
        )

    def _normalize_timestep(
        self,
        timestep: int | Tensor,
        batch: FactorGraphBatch,
    ) -> Tensor:
        value = torch.as_tensor(
            timestep,
            dtype=torch.long,
            device=batch.candidate_mask.device,
        )
        if value.ndim == 0:
            value = value.expand(batch.batch_size)
        if value.shape != (batch.batch_size,):
            raise ValueError("timestep must be scalar or have shape [B].")
        if (value < 1).any() or (value > self.config.num_mask_steps).any():
            raise ValueError("timestep is outside the configured MASK schedule.")
        return value

    def forward(
        self,
        batch: FactorGraphBatch,
        state: PartialPlacementState,
        timestep: int | Tensor,
    ) -> Tensor:
        """Return compatible `[B, M, D]` logits for clean device categories."""

        validate_partial_state(state, batch.candidate_mask, batch.service_mask)
        timestep_tensor = self._normalize_timestep(timestep, batch)
        graph = batch.graph
        hidden = {
            node_type: self.node_encoder[node_type](
                graph[node_type].x, graph[node_type].type_id
            )
            for node_type in self.node_types
        }
        dynamic = build_partial_context(batch, state)
        hidden["service"] = hidden["service"] + self.service_dynamic(
            dynamic["service"]
        )
        hidden["device"] = hidden["device"] + self.device_dynamic(dynamic["device"])
        hidden["dependency"] = hidden["dependency"] + self.dependency_dynamic(
            dynamic["dependency"]
        )

        service_nodes = batch.service_node_index[batch.service_mask]
        state_context = torch.zeros_like(hidden["service"])
        state_context.index_copy_(
            0,
            service_nodes,
            self.mask_embedding.unsqueeze(0).expand(service_nodes.numel(), -1),
        )
        if state.committed_mask.any():
            selected_device_nodes = batch.device_node_index.gather(
                1, state.assignment.clamp_min(0)
            )
            committed_service_nodes = batch.service_node_index[state.committed_mask]
            committed_device_nodes = selected_device_nodes[state.committed_mask]
            state_context[committed_service_nodes] = self.selected_device(
                hidden["device"][committed_device_nodes]
            )
        service_graph_index = graph["service"].batch
        hidden["service"] = (
            hidden["service"]
            + state_context
            + self.timestep_embedding(timestep_tensor[service_graph_index])
        )

        for layer in self.layers:
            hidden = layer(hidden, graph)

        service_dense = hidden["service"][batch.service_node_index.clamp_min(0)]
        device_dense = hidden["device"][batch.device_node_index.clamp_min(0)]
        choice_hidden = (
            self.choice_service(service_dense).unsqueeze(2)
            + self.choice_device(device_dense).unsqueeze(1)
            + self.choice_processing(batch.processing_latency.unsqueeze(-1))
        )
        logits = self.choice_output(torch.nn.functional.gelu(choice_hidden)).squeeze(-1)
        return logits.masked_fill(~batch.candidate_mask, -torch.inf)
