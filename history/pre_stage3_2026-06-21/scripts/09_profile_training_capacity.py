"""Profile a relation-heavy proxy for the planned typed factor denoiser."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from gdm_factor_diffusion.data import (
    build_factor_graph_blueprint,
    load_manifest,
    load_manifest_instance,
)


NODE_TYPES = ("service", "device", "dependency", "application")


@dataclass(frozen=True, slots=True)
class ProfileCase:
    name: str
    graph_copies: int
    hidden_dim: int
    layers: int


class ProxyLayer(nn.Module):
    def __init__(self, hidden_dim: int, relations: tuple[str, ...]) -> None:
        super().__init__()
        self.self_linear = nn.ModuleDict(
            {node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in NODE_TYPES}
        )
        self.relation_linear = nn.ModuleDict(
            {
                relation.replace("__", "_"): nn.Linear(hidden_dim, hidden_dim)
                for relation in relations
            }
        )
        self.relations = relations

    def forward(
        self,
        features: dict[str, torch.Tensor],
        relation_index: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        output = {
            node_type: self.self_linear[node_type](features[node_type])
            for node_type in NODE_TYPES
        }
        for relation in self.relations:
            source_type, _, target_type = relation.split("__")
            edge_index = relation_index[relation]
            message = self.relation_linear[relation.replace("__", "_")](
                features[source_type][edge_index[0]]
            )
            output[target_type] = output[target_type].index_add(
                0, edge_index[1], message
            )
        return {node_type: torch.nn.functional.gelu(value) for node_type, value in output.items()}


class ProxyDenoiser(nn.Module):
    def __init__(self, hidden_dim: int, layers: int, relations: tuple[str, ...]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            ProxyLayer(hidden_dim, relations) for _ in range(layers)
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        relation_index: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        for layer in self.layers:
            features = layer(features, relation_index)
        return features


def _largest_blueprint(dataset_root: Path):
    manifest = load_manifest(dataset_root / "manifest.json")
    instances = [
        load_manifest_instance(dataset_root, entry) for entry in manifest["instances"]
    ]
    instance = max(
        instances,
        key=lambda item: sum(
            index.shape[1]
            for index in build_factor_graph_blueprint(item).relation_index.values()
        ),
    )
    return instance, build_factor_graph_blueprint(instance)


def _repeat_graph(blueprint, copies: int, device: torch.device):
    node_counts = blueprint.node_counts
    relation_index: dict[str, torch.Tensor] = {}
    for relation, raw_index in blueprint.relation_index.items():
        source_type, _, target_type = relation.split("__")
        base = torch.as_tensor(raw_index, dtype=torch.long, device=device)
        repeated = []
        for copy in range(copies):
            offset = torch.tensor(
                [
                    [copy * node_counts[source_type]],
                    [copy * node_counts[target_type]],
                ],
                dtype=torch.long,
                device=device,
            )
            repeated.append(base + offset)
        relation_index[relation] = torch.cat(repeated, dim=1)
    return relation_index


def _profile_case(blueprint, case: ProfileCase, steps: int) -> dict[str, float | int | str]:
    device = torch.device("cuda")
    relations = tuple(blueprint.relation_index)
    relation_index = _repeat_graph(blueprint, case.graph_copies, device)
    model = ProxyDenoiser(case.hidden_dim, case.layers, relations).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = {
        node_type: torch.randn(
            count * case.graph_copies,
            case.hidden_dim,
            device=device,
        )
        for node_type, count in blueprint.node_counts.items()
    }

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(features, relation_index)
            loss = sum(value.square().mean() for value in output.values())
        loss.backward()
        optimizer.step()

    train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        train_step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "name": case.name,
        "graph_copies": case.graph_copies,
        "hidden_dim": case.hidden_dim,
        "layers": case.layers,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "milliseconds_per_step": elapsed * 1000 / steps,
    }
    del model, optimizer, features, relation_index
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=None,
        help="Dataset root containing manifest.json.",
    )
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training-capacity profile.")
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    root = args.dataset_root or (
        implementation_root / "artifacts" / "datasets" / "phase1b-smoke"
    )
    instance, blueprint = _largest_blueprint(root)
    cases = (
        ProfileCase("development", graph_copies=16, hidden_dim=128, layers=4),
        ProfileCase("comfortable", graph_copies=64, hidden_dim=128, layers=4),
        ProfileCase("large_hidden", graph_copies=32, hidden_dim=256, layers=6),
        ProfileCase("stress", graph_copies=128, hidden_dim=256, layers=6),
        ProfileCase("upper_bound", graph_copies=256, hidden_dim=512, layers=8),
    )
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "largest_instance": instance.instance_id,
        "largest_node_counts": blueprint.node_counts,
        "largest_relation_edges": sum(
            index.shape[1] for index in blueprint.relation_index.values()
        ),
        "cases": [_profile_case(blueprint, case, args.steps) for case in cases],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
