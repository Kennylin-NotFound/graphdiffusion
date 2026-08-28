"""Service-order helpers for sequential deployment policies."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from gdm_factor_diffusion.data.schema import DeploymentInstance


def service_order(instance: DeploymentInstance) -> np.ndarray:
    """Return a deterministic active-service order for one instance."""

    order = np.asarray(instance.topological_order, dtype=np.int64)
    expected = np.arange(instance.num_services, dtype=np.int64)
    if order.shape != expected.shape or set(order.tolist()) != set(expected.tolist()):
        order = expected
    return order.copy()


def service_order_batch(
    instances: Sequence[DeploymentInstance],
    *,
    max_services: int | None = None,
) -> Tensor:
    """Build a padded `[B, M]` service-order tensor with `-1` padding."""

    if not instances:
        raise ValueError("At least one instance is required.")
    width = max_services or max(instance.num_services for instance in instances)
    order = torch.full((len(instances), width), -1, dtype=torch.long)
    for batch_index, instance in enumerate(instances):
        values = torch.from_numpy(service_order(instance))
        order[batch_index, : values.numel()] = values
    return order
