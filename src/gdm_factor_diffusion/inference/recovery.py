"""Proposal-conditioned feasibility recovery for categorical placements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gdm_factor_diffusion.data.schema import DeploymentInstance
from gdm_factor_diffusion.solver.pair_costs import DependencyPairCosts, build_dependency_pair_costs
from gdm_factor_diffusion.solver.placement_verifier import (
    PlacementVerification,
    verify_placement,
)


@dataclass(frozen=True, slots=True)
class ProposalRecoveryConfig:
    """Bound the number of proposal assignments that recovery may replace."""

    max_released_services: int = 4
    tolerance: float = 1e-8

    def validate(self) -> None:
        if self.max_released_services < 0:
            raise ValueError("max_released_services must be nonnegative.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")


@dataclass(frozen=True, slots=True)
class ProposalRecoveryResult:
    original_placement: np.ndarray
    placement: np.ndarray | None
    success: bool
    preserved_services: tuple[int, ...]
    released_services: tuple[int, ...]
    completion_order: tuple[int, ...]
    initial_verification: PlacementVerification
    final_verification: PlacementVerification | None
    failure_reason: str | None


def _incident_dependencies(instance: DeploymentInstance) -> tuple[tuple[int, ...], ...]:
    incident: list[list[int]] = [[] for _ in range(instance.num_services)]
    for edge, (source, target) in enumerate(instance.dependency_index.T):
        incident[int(source)].append(edge)
        incident[int(target)].append(edge)
    return tuple(tuple(edges) for edges in incident)


def _admissible_devices(
    instance: DeploymentInstance,
    selected: np.ndarray,
    load: np.ndarray,
    service: int,
    incident: tuple[tuple[int, ...], ...],
    pair_costs: DependencyPairCosts,
    *,
    tolerance: float,
) -> tuple[int, ...]:
    """Return choices satisfying the same incremental hard conditions as decoding."""

    options: list[int] = []
    demand = instance.service_demand[service].astype(np.float64)
    for device_value in np.flatnonzero(instance.compatibility_mask[service]):
        device = int(device_value)
        if (
            load[device] + demand - instance.device_capacity[device]
            > tolerance
        ).any():
            continue
        admissible = True
        for edge in incident[service]:
            source, target = instance.dependency_index[:, edge]
            source = int(source)
            target = int(target)
            other = target if source == service else source
            other_device = int(selected[other])
            if other_device < 0:
                continue
            source_device = device if source == service else other_device
            target_device = device if target == service else other_device
            if not pair_costs.admissible[edge, source_device, target_device]:
                admissible = False
                break
        if admissible:
            options.append(device)
    return tuple(options)


def _proposal_scores(
    instance: DeploymentInstance,
    proposal: np.ndarray,
    model_probability: np.ndarray | None,
) -> np.ndarray:
    """Build deterministic device scores while retaining proposal preference."""

    if model_probability is None:
        scores = np.zeros(
            (instance.num_services, instance.num_devices), dtype=np.float64
        )
        valid = (proposal >= 0) & (proposal < instance.num_devices)
        services = np.flatnonzero(valid)
        scores[services, proposal[services]] = 1.0
        return scores

    scores = np.asarray(model_probability, dtype=np.float64)
    expected = (instance.num_services, instance.num_devices)
    if scores.shape != expected:
        raise ValueError(f"model_probability must have shape {expected}.")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("model_probability must be finite and nonnegative.")
    return scores.copy()


def recover_from_proposal(
    instance: DeploymentInstance,
    placement: np.ndarray,
    *,
    model_probability: np.ndarray | None = None,
    config: ProposalRecoveryConfig | None = None,
) -> ProposalRecoveryResult:
    """Preserve a hard-feasible proposal subset and greedily re-complete it.

    Proposed assignments are considered in descending model confidence. An
    assignment is preserved only if adding it to the current partial placement
    satisfies compatibility, residual capacity, and links to visible neighbors.
    The released services are then completed with the same incremental masks and
    model scores. Recovery has no independent search or latency heuristic.
    """

    settings = config or ProposalRecoveryConfig()
    settings.validate()
    raw = np.asarray(placement)
    initial = verify_placement(instance, raw, tolerance=settings.tolerance)
    proposal = initial.placement.copy()
    scores = _proposal_scores(instance, proposal, model_probability)
    pair_costs = build_dependency_pair_costs(instance)
    incident = _incident_dependencies(instance)
    selected = np.full(instance.num_services, -1, dtype=np.int64)
    load = np.zeros_like(instance.device_capacity, dtype=np.float64)

    proposed_confidence = np.full(instance.num_services, -1.0, dtype=np.float64)
    valid_proposed = (proposal >= 0) & (proposal < instance.num_devices)
    services = np.flatnonzero(valid_proposed)
    proposed_confidence[services] = scores[services, proposal[services]]
    preserve_order = sorted(
        range(instance.num_services),
        key=lambda service: (-proposed_confidence[service], service),
    )

    preserved: list[int] = []
    released: list[int] = []
    for service in preserve_order:
        proposed_device = int(proposal[service])
        options = _admissible_devices(
            instance,
            selected,
            load,
            service,
            incident,
            pair_costs,
            tolerance=settings.tolerance,
        )
        if proposed_device in options:
            selected[service] = proposed_device
            load[proposed_device] += instance.service_demand[service]
            preserved.append(service)
        else:
            released.append(service)

    if len(released) > settings.max_released_services:
        return ProposalRecoveryResult(
            original_placement=proposal,
            placement=None,
            success=False,
            preserved_services=tuple(sorted(preserved)),
            released_services=tuple(sorted(released)),
            completion_order=(),
            initial_verification=initial,
            final_verification=None,
            failure_reason="release_budget_exceeded",
        )

    completion_order: list[int] = []
    unresolved = set(released)
    while unresolved:
        choices: list[tuple[float, int, tuple[int, ...]]] = []
        for service in sorted(unresolved):
            options = _admissible_devices(
                instance,
                selected,
                load,
                service,
                incident,
                pair_costs,
                tolerance=settings.tolerance,
            )
            if not options:
                return ProposalRecoveryResult(
                    original_placement=proposal,
                    placement=None,
                    success=False,
                    preserved_services=tuple(sorted(preserved)),
                    released_services=tuple(sorted(released)),
                    completion_order=tuple(completion_order),
                    initial_verification=initial,
                    final_verification=None,
                    failure_reason="no_residual_candidate",
                )
            best_score = max(float(scores[service, device]) for device in options)
            choices.append((-best_score, service, options))

        _, service, options = min(choices)
        proposed_device = int(proposal[service])
        device = min(
            options,
            key=lambda candidate: (
                -float(scores[service, candidate]),
                candidate != proposed_device,
                candidate,
            ),
        )
        selected[service] = device
        load[device] += instance.service_demand[service]
        unresolved.remove(service)
        completion_order.append(service)

    final = verify_placement(instance, selected, tolerance=settings.tolerance)
    return ProposalRecoveryResult(
        original_placement=proposal,
        placement=selected.copy() if final.feasible else None,
        success=final.feasible,
        preserved_services=tuple(sorted(preserved)),
        released_services=tuple(sorted(released)),
        completion_order=tuple(completion_order),
        initial_verification=initial,
        final_verification=final,
        failure_reason=None if final.feasible else "final_verification_failed",
    )

