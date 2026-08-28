"""Bounded exhaustive enumeration for small-instance ground truth."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from gdm_factor_diffusion.data.schema import DeploymentInstance

from .latency_evaluator import evaluate_latency
from .placement_verifier import verify_placement


@dataclass(frozen=True, slots=True)
class ExhaustiveResult:
    placements: np.ndarray
    objectives: np.ndarray
    num_candidates: int
    num_feasible: int

    @property
    def best_placement(self) -> np.ndarray:
        if self.num_feasible == 0:
            raise ValueError("No feasible placement was found.")
        return self.placements[0].copy()

    @property
    def best_objective(self) -> float:
        if self.num_feasible == 0:
            raise ValueError("No feasible placement was found.")
        return float(self.objectives[0])


def enumerate_feasible_placements(
    instance: DeploymentInstance,
    *,
    max_states: int = 1_000_000,
) -> ExhaustiveResult:
    """Enumerate compatible categorical states and retain feasible placements."""

    candidate_devices = [
        np.flatnonzero(instance.compatibility_mask[service]).tolist()
        for service in range(instance.num_services)
    ]
    num_candidates = math.prod(len(candidates) for candidates in candidate_devices)
    if num_candidates > max_states:
        raise ValueError(
            f"Exhaustive enumeration requires {num_candidates} states, "
            f"exceeding max_states={max_states}."
        )

    feasible: list[tuple[float, tuple[int, ...]]] = []
    for state in itertools.product(*candidate_devices):
        placement = np.asarray(state, dtype=np.int64)
        if verify_placement(instance, placement).feasible:
            feasible.append((evaluate_latency(instance, placement).objective, state))
    feasible.sort(key=lambda item: (item[0], item[1]))

    if feasible:
        objectives = np.asarray([item[0] for item in feasible], dtype=np.float64)
        placements = np.asarray([item[1] for item in feasible], dtype=np.int64)
    else:
        objectives = np.empty((0,), dtype=np.float64)
        placements = np.empty((0, instance.num_services), dtype=np.int64)
    return ExhaustiveResult(
        placements=placements,
        objectives=objectives,
        num_candidates=num_candidates,
        num_feasible=len(feasible),
    )
