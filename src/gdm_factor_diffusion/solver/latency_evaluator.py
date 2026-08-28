"""Exact deterministic evaluation of the paper's end-to-end latency objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gdm_factor_diffusion.data.schema import DeploymentInstance

from .pair_costs import build_dependency_pair_costs
from .placement_verifier import PlacementVerification, verify_placement


class InfeasiblePlacementError(ValueError):
    def __init__(self, verification: PlacementVerification) -> None:
        super().__init__(
            "Latency is defined only for feasible placements; "
            f"found {verification.num_violations} violation(s)."
        )
        self.verification = verification


@dataclass(frozen=True, slots=True)
class LatencyEvaluation:
    placement: np.ndarray
    processing_latency: np.ndarray
    transmission_latency: np.ndarray
    completion_time: np.ndarray
    critical_predecessor_edge: np.ndarray
    application_latency: np.ndarray
    critical_sink: np.ndarray
    objective: float


def evaluate_latency(
    instance: DeploymentInstance,
    placement: np.ndarray,
) -> LatencyEvaluation:
    """Evaluate one verified placement on the merged joint DAG."""

    verification = verify_placement(instance, placement)
    if not verification.feasible:
        raise InfeasiblePlacementError(verification)
    selected = verification.placement

    service = np.arange(instance.num_services)
    processing = instance.processing_latency[service, selected].astype(np.float64)

    pair_costs = build_dependency_pair_costs(instance)
    pair_admissible, transmission = pair_costs.selected(
        selected, instance.dependency_index
    )
    if not pair_admissible.all():
        raise RuntimeError("Verifier and dependency pair-cost mask disagree.")

    incoming: list[list[int]] = [[] for _ in range(instance.num_services)]
    for edge, target in enumerate(instance.dependency_index[1]):
        incoming[int(target)].append(edge)

    completion = np.zeros(instance.num_services, dtype=np.float64)
    critical_predecessor_edge = np.full(instance.num_services, -1, dtype=np.int64)
    source_index = instance.dependency_index[0]
    for target in instance.topological_order:
        target = int(target)
        predecessor_edges = incoming[target]
        if predecessor_edges:
            arrivals = np.asarray(
                [
                    completion[int(source_index[edge])] + transmission[edge]
                    for edge in predecessor_edges
                ],
                dtype=np.float64,
            )
            critical_local = int(np.argmax(arrivals))
            critical_predecessor_edge[target] = predecessor_edges[critical_local]
            completion[target] = processing[target] + arrivals[critical_local]
        else:
            completion[target] = processing[target]

    application_latency = np.zeros(instance.num_applications, dtype=np.float64)
    critical_sink = np.full(instance.num_applications, -1, dtype=np.int64)
    for application in range(instance.num_applications):
        sinks = np.flatnonzero(instance.sink_mask[application])
        sink_completion = completion[sinks]
        critical_local = int(np.argmax(sink_completion))
        critical_sink[application] = int(sinks[critical_local])
        application_latency[application] = sink_completion[critical_local]

    objective = float(
        np.dot(instance.application_weight.astype(np.float64), application_latency)
    )
    return LatencyEvaluation(
        placement=selected.copy(),
        processing_latency=processing,
        transmission_latency=transmission.copy(),
        completion_time=completion,
        critical_predecessor_edge=critical_predecessor_edge,
        application_latency=application_latency,
        critical_sink=critical_sink,
        objective=objective,
    )
