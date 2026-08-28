"""Finite device-pair costs and masks for dependency factors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gdm_factor_diffusion.data.schema import DeploymentInstance


@dataclass(frozen=True, slots=True)
class DependencyPairCosts:
    """Graph-ready direct-link feasibility and finite transmission costs."""

    admissible: np.ndarray
    transmission_latency: np.ndarray

    def selected(
        self,
        placement: np.ndarray,
        dependency_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_device = placement[dependency_index[0]]
        target_device = placement[dependency_index[1]]
        edge = np.arange(dependency_index.shape[1])
        return (
            self.admissible[edge, source_device, target_device],
            self.transmission_latency[edge, source_device, target_device],
        )


def build_dependency_pair_costs(instance: DeploymentInstance) -> DependencyPairCosts:
    """Build finite pair costs while keeping infeasibility in a separate mask."""

    e = instance.num_dependencies
    d = instance.num_devices
    latency = np.zeros((e, d, d), dtype=np.float64)

    colocated = np.eye(d, dtype=np.bool_)
    physical_admissible = colocated | instance.connectivity
    if e == 0:
        return DependencyPairCosts(
            admissible=np.zeros((0, d, d), dtype=np.bool_),
            transmission_latency=latency,
        )

    source, target = instance.dependency_index
    endpoint_compatible = (
        instance.compatibility_mask[source, :, None]
        & instance.compatibility_mask[target, None, :]
    )
    admissible = endpoint_compatible & physical_admissible[None, :, :]

    direct_link = instance.connectivity
    for edge in range(e):
        latency[edge, direct_link] = (
            float(instance.dependency_data_volume[edge])
            / instance.link_rate[direct_link].astype(np.float64)
        )

    if not np.isfinite(latency).all():
        raise ValueError("Dependency pair costs must remain finite.")
    if not (latency[:, colocated] == 0).all():
        raise ValueError("Colocated dependency pair costs must be zero.")

    return DependencyPairCosts(
        admissible=np.ascontiguousarray(admissible),
        transmission_latency=np.ascontiguousarray(latency),
    )
