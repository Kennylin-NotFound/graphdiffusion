"""Minimal typed factor-graph clean-state predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from gdm_factor_diffusion.data.catalogs import DEVICE_TYPES, SERVICE_TYPES
from gdm_factor_diffusion.diffusion.masking import validate_state
from gdm_factor_diffusion.graph.batch_adapter import (
    FactorGraphBatch,
    build_dynamic_context,
)

from .factor_layer import TypedFactorLayer


@dataclass(frozen=True, slots=True)
class DenoiserConfig:
    num_diffusion_steps: int = 100
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        if self.num_diffusion_steps < 1:
            raise ValueError("num_diffusion_steps must be positive.")
        if self.hidden_dim < 1 or self.num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")


@dataclass(frozen=True, slots=True)
class DirectPredictorConfig:
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        if self.hidden_dim < 1 or self.num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")


class _NodeEncoder(nn.Module):
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
        return torch.nn.functional.gelu(
            self.normalization(encoded)
        )


class TypedFactorDenoiser(nn.Module):
    """Predict compatible-device logits for the clean categorical placement."""

    def __init__(
        self,
        *,
        node_feature_dims: dict[str, int],
        node_type_counts: dict[str, int | None],
        edge_types: tuple[tuple[str, str, str], ...],
        resource_dim: int,
        config: DenoiserConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DenoiserConfig()
        self.config.validate()
        hidden_dim = self.config.hidden_dim
        self.node_types = tuple(node_feature_dims)
        self.edge_types = edge_types
        self.node_encoder = nn.ModuleDict(
            {
                node_type: _NodeEncoder(
                    node_feature_dims[node_type],
                    node_type_counts[node_type],
                    hidden_dim,
                )
                for node_type in self.node_types
            }
        )
        self.timestep_embedding = nn.Embedding(
            self.config.num_diffusion_steps + 1, hidden_dim
        )
        self.service_dynamic = nn.Linear(1, hidden_dim)
        self.device_dynamic = nn.Linear(resource_dim, hidden_dim)
        self.dependency_dynamic = nn.Linear(2, hidden_dim)
        self.selected_device = nn.Linear(hidden_dim, hidden_dim, bias=False)
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
        config: DenoiserConfig | None = None,
    ) -> "TypedFactorDenoiser":
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
            raise ValueError("timestep must be a scalar or a vector with shape [B].")
        if (value < 1).any() or (value > self.config.num_diffusion_steps).any():
            raise ValueError("timestep is outside the configured diffusion schedule.")
        return value

    def forward(
        self,
        batch: FactorGraphBatch,
        noisy_state: Tensor,
        timestep: int | Tensor,
    ) -> Tensor:
        """Return `[B, M, D]` clean-state logits with invalid entries at `-inf`."""

        validate_state(noisy_state, batch.candidate_mask, batch.service_mask)
        timestep_tensor = self._normalize_timestep(timestep, batch)
        graph = batch.graph
        hidden = {
            node_type: self.node_encoder[node_type](
                graph[node_type].x, graph[node_type].type_id
            )
            for node_type in self.node_types
        }

        dynamic = build_dynamic_context(batch, noisy_state)
        hidden["service"] = hidden["service"] + self.service_dynamic(dynamic["service"])
        hidden["device"] = hidden["device"] + self.device_dynamic(dynamic["device"])
        hidden["dependency"] = hidden["dependency"] + self.dependency_dynamic(
            dynamic["dependency"]
        )

        service_node = batch.service_node_index[batch.service_mask]
        selected_device_node = batch.selected_device_node_index(noisy_state)[
            batch.service_mask
        ]
        selected_context = torch.zeros_like(hidden["service"])
        selected_context[service_node] = self.selected_device(
            hidden["device"][selected_device_node]
        )
        service_graph_index = graph["service"].batch
        hidden["service"] = (
            hidden["service"]
            + selected_context
            + self.timestep_embedding(timestep_tensor[service_graph_index])
        )

        for layer in self.layers:
            hidden = layer(hidden, graph)

        service_dense = hidden["service"][
            batch.service_node_index.clamp_min(0)
        ]
        device_dense = hidden["device"][
            batch.device_node_index.clamp_min(0)
        ]
        choice_hidden = (
            self.choice_service(service_dense).unsqueeze(2)
            + self.choice_device(device_dense).unsqueeze(1)
            + self.choice_processing(batch.processing_latency.unsqueeze(-1))
        )
        logits = self.choice_output(torch.nn.functional.gelu(choice_hidden)).squeeze(-1)
        return logits.masked_fill(~batch.candidate_mask, -torch.inf)


class TypedFactorDirectPredictor(nn.Module):
    """Predict clean placement logits from the static typed factor graph."""

    def __init__(
        self,
        *,
        node_feature_dims: dict[str, int],
        node_type_counts: dict[str, int | None],
        edge_types: tuple[tuple[str, str, str], ...],
        config: DirectPredictorConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DirectPredictorConfig()
        self.config.validate()
        hidden_dim = self.config.hidden_dim
        self.node_types = tuple(node_feature_dims)
        self.edge_types = edge_types
        self.node_encoder = nn.ModuleDict(
            {
                node_type: _NodeEncoder(
                    node_feature_dims[node_type],
                    node_type_counts[node_type],
                    hidden_dim,
                )
                for node_type in self.node_types
            }
        )
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
        config: DirectPredictorConfig | None = None,
    ) -> "TypedFactorDirectPredictor":
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
            config=config,
        )

    def forward(self, batch: FactorGraphBatch) -> Tensor:
        """Return `[B, M, D]` clean-state logits with invalid entries at `-inf`."""

        graph = batch.graph
        hidden = {
            node_type: self.node_encoder[node_type](
                graph[node_type].x, graph[node_type].type_id
            )
            for node_type in self.node_types
        }
        for layer in self.layers:
            hidden = layer(hidden, graph)

        service_dense = hidden["service"][
            batch.service_node_index.clamp_min(0)
        ]
        device_dense = hidden["device"][
            batch.device_node_index.clamp_min(0)
        ]
        choice_hidden = (
            self.choice_service(service_dense).unsqueeze(2)
            + self.choice_device(device_dense).unsqueeze(1)
            + self.choice_processing(batch.processing_latency.unsqueeze(-1))
        )
        logits = self.choice_output(torch.nn.functional.gelu(choice_hidden)).squeeze(-1)
        return logits.masked_fill(~batch.candidate_mask, -torch.inf)
