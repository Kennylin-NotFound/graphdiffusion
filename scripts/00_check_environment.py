"""Validate the Phase 0 runtime without modifying external state."""

from __future__ import annotations

import json
import platform
import sys
from importlib import import_module, util


def package_version(module_name: str) -> str:
    module = import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def check_torch() -> dict[str, object]:
    import torch

    result: dict[str, object] = {
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    device = torch.device("cuda")
    x = torch.randn(512, 128, device=device, requires_grad=True)
    weight = torch.randn(128, 64, device=device, requires_grad=True)
    loss = (x @ weight).square().mean()
    loss.backward()

    if not torch.isfinite(x.grad).all() or not torch.isfinite(weight.grad).all():
        raise RuntimeError("CUDA backward pass produced non-finite gradients.")

    result.update(
        {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_backward_ok": True,
        }
    )
    return result


def check_pyg() -> dict[str, object]:
    import torch
    import torch_geometric
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
    from torch_geometric.typing import (
        WITH_PYG_LIB,
        WITH_TORCH_SCATTER,
        WITH_TORCH_SPARSE,
    )

    graph = HeteroData()
    graph["service"].x = torch.randn(4, 8)
    graph["device"].x = torch.randn(3, 8)
    graph["service", "compatible_with", "device"].edge_index = torch.tensor(
        [[0, 0, 1, 2, 3], [0, 1, 1, 2, 0]]
    )
    graph["device", "hosts", "service"].edge_index = graph[
        "service", "compatible_with", "device"
    ].edge_index.flip(0)
    graph = graph.to("cuda")

    layer = HeteroConv(
        {
            ("service", "compatible_with", "device"): SAGEConv((-1, -1), 16),
            ("device", "hosts", "service"): SAGEConv((-1, -1), 16),
        },
        aggr="sum",
    ).to("cuda")
    output = layer(graph.x_dict, graph.edge_index_dict)
    loss = sum(features.square().mean() for features in output.values())
    loss.backward()

    if not all(parameter.grad is not None for parameter in layer.parameters()):
        raise RuntimeError("PyG heterogeneous backward pass did not produce gradients.")

    return {
        "version": torch_geometric.__version__,
        "heterogeneous_forward_backward_ok": True,
        "output_shapes": {
            node_type: list(features.shape)
            for node_type, features in output.items()
        },
        "optional_extensions": {
            "pyg_lib": WITH_PYG_LIB,
            "torch_scatter": WITH_TORCH_SCATTER,
            "torch_sparse": WITH_TORCH_SPARSE,
        },
    }


def check_gurobi() -> dict[str, object]:
    import gurobipy as gp

    result: dict[str, object] = {"version": list(gp.gurobi.version())}
    try:
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 0)
        environment.start()
        model = gp.Model(env=environment)
        model.dispose()
        environment.dispose()
        result["license_ok_in_current_process"] = True
    except gp.GurobiError as error:
        result["license_ok_in_current_process"] = False
        result["license_error"] = str(error)
    return result


def main() -> None:
    report = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "platform": platform.platform(),
        },
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "scipy",
                "pandas",
                "networkx",
                "yaml",
                "pytest",
                "wandb",
                "tensorboard",
            )
        },
        "torch": check_torch(),
        "pyg": check_pyg(),
        "gurobi": check_gurobi(),
        "optional_module_specs": {
            name: util.find_spec(name) is not None
            for name in ("pyg_lib", "torch_scatter", "torch_sparse")
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
