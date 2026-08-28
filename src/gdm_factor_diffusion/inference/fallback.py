"""Deterministic constructive fallback with bounded backtracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gdm_factor_diffusion.data.schema import DeploymentInstance
from gdm_factor_diffusion.solver.pair_costs import build_dependency_pair_costs
from gdm_factor_diffusion.solver.placement_verifier import (
    PlacementVerification,
    verify_placement,
)


@dataclass(frozen=True, slots=True)
class ConstructiveFallbackConfig:
    max_search_nodes: int = 100_000
    tolerance: float = 1e-8

    def validate(self) -> None:
        if self.max_search_nodes < 1:
            raise ValueError("max_search_nodes must be positive.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")


@dataclass(frozen=True, slots=True)
class ConstructiveFallbackResult:
    placement: np.ndarray | None
    success: bool
    search_nodes: int
    exhausted_budget: bool
    verification: PlacementVerification | None


def construct_feasible_placement(
    instance: DeploymentInstance,
    *,
    config: ConstructiveFallbackConfig | None = None,
) -> ConstructiveFallbackResult:
    """Find a feasible placement using deterministic MRV-ordered backtracking."""

    settings = config or ConstructiveFallbackConfig()
    settings.validate()
    m = instance.num_services
    r = instance.num_resources
    selected = np.full(m, -1, dtype=np.int64)
    load = np.zeros((instance.num_devices, r), dtype=np.float64)
    pair_costs = build_dependency_pair_costs(instance)
    incident: list[list[int]] = [[] for _ in range(m)]
    for edge, (source, target) in enumerate(instance.dependency_index.T):
        incident[int(source)].append(edge)
        incident[int(target)].append(edge)
    search_nodes = 0
    exhausted = False

    def feasible_devices(service: int) -> list[tuple[float, int]]:
        options: list[tuple[float, int]] = []
        demand = instance.service_demand[service].astype(np.float64)
        for device in np.flatnonzero(instance.compatibility_mask[service]):
            device = int(device)
            projected = load[device] + demand
            if (projected - instance.device_capacity[device] > settings.tolerance).any():
                continue
            local_cost = float(instance.processing_latency[service, device])
            valid = True
            for edge in incident[service]:
                source, target = instance.dependency_index[:, edge]
                other = int(target if int(source) == service else source)
                other_device = int(selected[other])
                if other_device < 0:
                    continue
                source_device = device if int(source) == service else other_device
                target_device = device if int(target) == service else other_device
                if not pair_costs.admissible[edge, source_device, target_device]:
                    valid = False
                    break
                local_cost += float(
                    pair_costs.transmission_latency[edge, source_device, target_device]
                )
            if not valid:
                continue
            utilization = projected / np.maximum(
                instance.device_capacity[device].astype(np.float64),
                settings.tolerance,
            )
            options.append((local_cost + 0.01 * float(utilization.sum()), device))
        options.sort()
        return options

    def search(assigned_count: int) -> bool:
        nonlocal search_nodes, exhausted
        if assigned_count == m:
            return verify_placement(
                instance, selected, tolerance=settings.tolerance
            ).feasible
        if search_nodes >= settings.max_search_nodes:
            exhausted = True
            return False

        choice: tuple[int, int, int, list[tuple[float, int]]] | None = None
        for service in np.flatnonzero(selected < 0):
            service = int(service)
            options = feasible_devices(service)
            if not options:
                return False
            key = (len(options), -len(incident[service]), service, options)
            if choice is None or key[:3] < choice[:3]:
                choice = key
        assert choice is not None
        service = choice[2]
        demand = instance.service_demand[service].astype(np.float64)
        for _, device in choice[3]:
            if search_nodes >= settings.max_search_nodes:
                exhausted = True
                return False
            search_nodes += 1
            selected[service] = device
            load[device] += demand
            if search(assigned_count + 1):
                return True
            load[device] -= demand
            selected[service] = -1
        return False

    success = search(0)
    placement = selected.copy() if success else None
    verification = (
        verify_placement(instance, placement, tolerance=settings.tolerance)
        if placement is not None
        else None
    )
    return ConstructiveFallbackResult(
        placement=placement,
        success=bool(success and verification is not None and verification.feasible),
        search_nodes=search_nodes,
        exhausted_budget=exhausted,
        verification=verification,
    )
