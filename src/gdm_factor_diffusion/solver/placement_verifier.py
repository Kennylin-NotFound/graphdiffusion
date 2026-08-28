"""Detailed verification of categorical or binary placement representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gdm_factor_diffusion.data.schema import DeploymentInstance


@dataclass(frozen=True, slots=True)
class PlacementVerification:
    placement: np.ndarray
    feasible: bool
    assignment_valid: bool
    compatibility_valid: bool
    capacity_valid: bool
    direct_link_valid: bool
    format_error: str | None
    assignment_violations: tuple[int, ...]
    incompatible_services: tuple[int, ...]
    disconnected_dependencies: tuple[int, ...]
    capacity_load: np.ndarray
    capacity_excess: np.ndarray

    @property
    def total_capacity_excess(self) -> float:
        return float(self.capacity_excess.sum())

    @property
    def num_violations(self) -> int:
        return (
            len(self.assignment_violations)
            + len(self.incompatible_services)
            + len(self.disconnected_dependencies)
            + int((self.capacity_excess > 0).sum())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "placement": self.placement.tolist(),
            "feasible": self.feasible,
            "assignment_valid": self.assignment_valid,
            "compatibility_valid": self.compatibility_valid,
            "capacity_valid": self.capacity_valid,
            "direct_link_valid": self.direct_link_valid,
            "format_error": self.format_error,
            "assignment_violations": list(self.assignment_violations),
            "incompatible_services": list(self.incompatible_services),
            "disconnected_dependencies": list(self.disconnected_dependencies),
            "capacity_load": self.capacity_load.tolist(),
            "capacity_excess": self.capacity_excess.tolist(),
        }


def _normalize_representation(
    instance: DeploymentInstance,
    placement: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], str | None]:
    raw = np.asarray(placement)
    m = instance.num_services
    d = instance.num_devices
    selected = np.full(m, -1, dtype=np.int64)

    if raw.shape == (m,):
        numeric = np.issubdtype(raw.dtype, np.number)
        if not numeric:
            return selected, tuple(range(m)), "Categorical placement must be numeric."
        finite = np.isfinite(raw)
        integer_like = finite & (raw == np.floor(raw))
        in_range = integer_like & (raw >= 0) & (raw < d)
        selected[in_range] = raw[in_range].astype(np.int64)
        invalid = tuple(np.flatnonzero(~in_range).tolist())
        return selected, invalid, None

    if raw.shape == (m, d):
        numeric = np.issubdtype(raw.dtype, np.number) or raw.dtype == np.bool_
        if not numeric:
            return selected, tuple(range(m)), "Binary placement matrix must be numeric."
        finite = np.isfinite(raw)
        binary = finite & ((raw == 0) | (raw == 1))
        row_valid = binary.all(axis=1) & (raw.sum(axis=1) == 1)
        selected[row_valid] = np.argmax(raw[row_valid], axis=1)
        invalid = tuple(np.flatnonzero(~row_valid).tolist())
        return selected, invalid, None

    return (
        selected,
        tuple(range(m)),
        f"Placement shape must be [M] or [M, D], received {raw.shape}.",
    )


def verify_placement(
    instance: DeploymentInstance,
    placement: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> PlacementVerification:
    """Check all constraints in the paper's deployment formulation."""

    selected, assignment_violations, format_error = _normalize_representation(
        instance, placement
    )
    assignment_valid = format_error is None and not assignment_violations
    assigned = selected >= 0

    service = np.flatnonzero(assigned)
    incompatible = tuple(
        service[~instance.compatibility_mask[service, selected[service]]].tolist()
    )
    compatibility_valid = not incompatible

    capacity_load = np.zeros_like(instance.device_capacity, dtype=np.float64)
    for service_index in service:
        capacity_load[selected[service_index]] += instance.service_demand[service_index]
    capacity_excess = np.maximum(
        capacity_load - instance.device_capacity.astype(np.float64),
        0.0,
    )
    capacity_valid = bool((capacity_excess <= tolerance).all())

    disconnected: list[int] = []
    for edge, (source, target) in enumerate(instance.dependency_index.T):
        source_device = selected[source]
        target_device = selected[target]
        if source_device < 0 or target_device < 0:
            continue
        if source_device != target_device and not instance.connectivity[
            source_device, target_device
        ]:
            disconnected.append(edge)
    direct_link_valid = not disconnected

    feasible = (
        assignment_valid
        and compatibility_valid
        and capacity_valid
        and direct_link_valid
    )
    return PlacementVerification(
        placement=selected,
        feasible=feasible,
        assignment_valid=assignment_valid,
        compatibility_valid=compatibility_valid,
        capacity_valid=capacity_valid,
        direct_link_valid=direct_link_valid,
        format_error=format_error,
        assignment_violations=assignment_violations,
        incompatible_services=incompatible,
        disconnected_dependencies=tuple(disconnected),
        capacity_load=capacity_load,
        capacity_excess=capacity_excess,
    )
