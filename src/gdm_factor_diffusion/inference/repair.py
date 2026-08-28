"""Deterministic bounded single-service repair for categorical placements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gdm_factor_diffusion.data.schema import DeploymentInstance
from gdm_factor_diffusion.solver.pair_costs import build_dependency_pair_costs
from gdm_factor_diffusion.solver.placement_verifier import (
    PlacementVerification,
    verify_placement,
)


@dataclass(frozen=True, order=True, slots=True)
class ViolationScore:
    """Lexicographic hard-violation score used to accept repair moves."""

    structural_violations: int
    hard_constraint_violations: int
    normalized_capacity_excess: float
    direct_link_conflicts: int


@dataclass(frozen=True, slots=True)
class RepairConfig:
    max_moves: int = 10
    tolerance: float = 1e-8

    def validate(self) -> None:
        if self.max_moves < 0:
            raise ValueError("max_moves must be nonnegative.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")


@dataclass(frozen=True, slots=True)
class RepairMove:
    service: int
    source_device: int
    target_device: int
    score_before: ViolationScore
    score_after: ViolationScore
    local_latency_cost: float
    negative_log_probability: float


@dataclass(frozen=True, slots=True)
class RepairResult:
    original_placement: np.ndarray
    placement: np.ndarray
    success: bool
    moves: tuple[RepairMove, ...]
    initial_verification: PlacementVerification
    final_verification: PlacementVerification


def violation_score(
    instance: DeploymentInstance,
    placement: np.ndarray,
    verification: PlacementVerification | None = None,
    *,
    tolerance: float = 1e-8,
) -> ViolationScore:
    """Summarize verifier output without replacing the final feasibility check."""

    report = verification or verify_placement(instance, placement, tolerance=tolerance)
    capacity_mask = report.capacity_excess > tolerance
    normalized_excess = report.capacity_excess / np.maximum(
        instance.device_capacity.astype(np.float64),
        tolerance,
    )
    structural = len(report.assignment_violations) + len(
        report.incompatible_services
    )
    link_conflicts = len(report.disconnected_dependencies)
    return ViolationScore(
        structural_violations=structural,
        hard_constraint_violations=int(capacity_mask.sum()) + link_conflicts,
        normalized_capacity_excess=float(normalized_excess.sum()),
        direct_link_conflicts=link_conflicts,
    )


def _affected_services(
    instance: DeploymentInstance,
    report: PlacementVerification,
    *,
    tolerance: float,
) -> tuple[int, ...]:
    selected = report.placement
    affected: set[int] = set(report.assignment_violations)
    affected.update(report.incompatible_services)
    overloaded_devices = np.flatnonzero(
        (report.capacity_excess > tolerance).any(axis=1)
    )
    for device in overloaded_devices:
        affected.update(np.flatnonzero(selected == device).tolist())
    for edge in report.disconnected_dependencies:
        affected.update(instance.dependency_index[:, edge].tolist())
    if not affected:
        affected.update(range(instance.num_services))
    return tuple(sorted(affected))


def _local_latency_cost(
    instance: DeploymentInstance,
    placement: np.ndarray,
    moved_service: int,
) -> float:
    """Return a deterministic local tie-break cost for an infeasible placement."""

    selected = np.asarray(placement, dtype=np.int64)
    device = int(selected[moved_service])
    cost = float(instance.processing_latency[moved_service, device])
    pair_costs = build_dependency_pair_costs(instance)
    for edge, (source, target) in enumerate(instance.dependency_index.T):
        if moved_service not in (int(source), int(target)):
            continue
        source_device = int(selected[source])
        target_device = int(selected[target])
        if pair_costs.admissible[edge, source_device, target_device]:
            cost += float(
                pair_costs.transmission_latency[edge, source_device, target_device]
            )
        else:
            cost += 1e6
    return cost


def repair_placement(
    instance: DeploymentInstance,
    placement: np.ndarray,
    *,
    model_probability: np.ndarray | None = None,
    config: RepairConfig | None = None,
) -> RepairResult:
    """Apply bounded moves that strictly reduce the lexicographic violation score."""

    settings = config or RepairConfig()
    settings.validate()
    original = np.asarray(placement, dtype=np.int64).copy()
    current = original.copy()
    initial = verify_placement(instance, current, tolerance=settings.tolerance)
    report = initial
    moves: list[RepairMove] = []

    probability = None
    if model_probability is not None:
        probability = np.asarray(model_probability, dtype=np.float64)
        if probability.shape != (instance.num_services, instance.num_devices):
            raise ValueError("model_probability must have shape [M, D].")

    for _ in range(settings.max_moves):
        if report.feasible:
            break
        before = violation_score(
            instance, current, report, tolerance=settings.tolerance
        )
        best_key: tuple | None = None
        best: tuple[np.ndarray, PlacementVerification, RepairMove] | None = None
        for service in _affected_services(
            instance, report, tolerance=settings.tolerance
        ):
            source_device = int(current[service])
            for target_device in np.flatnonzero(instance.compatibility_mask[service]):
                target_device = int(target_device)
                if target_device == source_device:
                    continue
                candidate = current.copy()
                candidate[service] = target_device
                candidate_report = verify_placement(
                    instance, candidate, tolerance=settings.tolerance
                )
                after = violation_score(
                    instance,
                    candidate,
                    candidate_report,
                    tolerance=settings.tolerance,
                )
                if not after < before:
                    continue
                local_cost = _local_latency_cost(instance, candidate, service)
                negative_log_probability = (
                    0.0
                    if probability is None
                    else float(
                        -np.log(max(probability[service, target_device], 1e-12))
                    )
                )
                key = (
                    after,
                    local_cost,
                    negative_log_probability,
                    service,
                    target_device,
                )
                move = RepairMove(
                    service=service,
                    source_device=source_device,
                    target_device=target_device,
                    score_before=before,
                    score_after=after,
                    local_latency_cost=local_cost,
                    negative_log_probability=negative_log_probability,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = candidate, candidate_report, move
        if best is None:
            break
        current, report, move = best
        moves.append(move)

    final = verify_placement(instance, current, tolerance=settings.tolerance)
    return RepairResult(
        original_placement=original,
        placement=current,
        success=final.feasible,
        moves=tuple(moves),
        initial_verification=initial,
        final_verification=final,
    )
