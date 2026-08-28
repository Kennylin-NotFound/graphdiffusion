"""Typed factor-graph construction and batching."""

from .batch_adapter import (
    FactorGraphBatch,
    GraphFeatureSchema,
    build_dynamic_context,
    build_factor_graph_batch,
    infer_feature_schema,
    merge_feature_schemas,
)
from .hetero_graph import build_hetero_graph
from .partial_context import build_partial_context

__all__ = [
    "FactorGraphBatch",
    "GraphFeatureSchema",
    "build_dynamic_context",
    "build_factor_graph_batch",
    "build_hetero_graph",
    "build_partial_context",
    "infer_feature_schema",
    "merge_feature_schemas",
]
