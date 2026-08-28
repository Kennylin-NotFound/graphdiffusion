"""Small residual message layer over typed factor-graph relations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


def relation_key(relation: tuple[str, str, str]) -> str:
    return "__".join(relation)


class TypedFactorLayer(nn.Module):
    """Mean-aggregate relation-specific messages with residual normalization."""

    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[tuple[str, str, str]],
        hidden_dim: int,
        *,
        edge_attr_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_types = tuple(node_types)
        self.edge_types = tuple(edge_types)
        self.self_linear = nn.ModuleDict(
            {node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types}
        )
        self.source_linear = nn.ModuleDict(
            {
                relation_key(edge_type): nn.Linear(hidden_dim, hidden_dim, bias=False)
                for edge_type in edge_types
            }
        )
        self.edge_linear = nn.ModuleDict(
            {
                relation_key(edge_type): nn.Linear(
                    edge_attr_dim, hidden_dim, bias=False
                )
                for edge_type in edge_types
            }
        )
        self.normalization = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in node_types}
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Mapping[str, Tensor],
        graph: object,
    ) -> dict[str, Tensor]:
        aggregate = {
            node_type: torch.zeros_like(hidden[node_type])
            for node_type in self.node_types
        }
        for edge_type in self.edge_types:
            source_type, _, target_type = edge_type
            key = relation_key(edge_type)
            edge_index = graph[edge_type].edge_index
            edge_attr = graph[edge_type].edge_attr
            message = (
                self.source_linear[key](hidden[source_type][edge_index[0]])
                + self.edge_linear[key](edge_attr)
            )
            relation_sum = torch.zeros_like(hidden[target_type])
            relation_sum.index_add_(0, edge_index[1], message)
            degree = torch.zeros(
                hidden[target_type].shape[0],
                dtype=message.dtype,
                device=message.device,
            )
            degree.index_add_(
                0,
                edge_index[1],
                torch.ones(edge_index.shape[1], dtype=message.dtype, device=message.device),
            )
            relation_sum = relation_sum / degree.clamp_min(1).unsqueeze(-1)
            aggregate[target_type] = aggregate[target_type] + relation_sum

        output: dict[str, Tensor] = {}
        for node_type in self.node_types:
            update = torch.nn.functional.gelu(
                self.self_linear[node_type](hidden[node_type])
                + aggregate[node_type]
            )
            output[node_type] = self.normalization[node_type](
                hidden[node_type] + self.dropout(update)
            )
        return output
