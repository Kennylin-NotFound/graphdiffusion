"""Independent optimization and heuristic baselines for scientific evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from gdm_factor_diffusion.data.schema import DeploymentInstance
from gdm_factor_diffusion.solver import (
    MilpConfig,
    build_dependency_pair_costs,
    evaluate_latency,
    solve_milp,
    verify_placement,
)

from .fallback import ConstructiveFallbackConfig, construct_feasible_placement
from .solve import SolveResult, VerifiedCandidate


@dataclass(frozen=True, slots=True)
class GreedyLocalResult:
    """One deterministic topological greedy construction attempt."""

    placement: np.ndarray | None
    success: bool
    assigned_services: int
    failed_service: int | None
    verification: Any | None


@dataclass(frozen=True, slots=True)
class LatencyAwareHeuristicCandidate:
    """One deterministic construction attempt used by the latency-aware heuristic."""

    placement: np.ndarray | None
    success: bool
    strategy: str
    scoring: str
    assigned_services: int
    failed_service: int | None
    objective: float | None
    verification: Any | None


@dataclass(frozen=True, slots=True)
class LatencyAwareHeuristicResult:
    """A compact deterministic portfolio selected by exact latency."""

    placement: np.ndarray | None
    success: bool
    objective: float | None
    selected_strategy: str | None
    selected_scoring: str | None
    candidates: tuple[LatencyAwareHeuristicCandidate, ...]


@dataclass(frozen=True, slots=True)
class LocalSearchConfig:
    """Budget for a deterministic one-service local search."""

    max_passes: int = 3
    max_evaluations: int = 600
    min_improvement: float = 1e-10
    fallback_max_search_nodes: int = 100_000
    tolerance: float = 1e-8

    def validate(self) -> None:
        if self.max_passes < 1 or self.max_evaluations < 1:
            raise ValueError("Local-search passes and evaluations must be positive.")
        if self.min_improvement < 0:
            raise ValueError("min_improvement must be nonnegative.")
        if self.fallback_max_search_nodes < 1:
            raise ValueError("fallback_max_search_nodes must be positive.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")


@dataclass(frozen=True, slots=True)
class LocalSearchResult:
    """Strictly improving local-search trace."""

    placement: np.ndarray | None
    success: bool
    objective: float | None
    initial_source: str | None
    initial_objective: float | None
    accepted_moves: int
    evaluated_moves: int
    passes_completed: int
    exhausted_budget: bool


def construct_greedy_local_placement(
    instance: DeploymentInstance,
    *,
    tolerance: float = 1e-8,
) -> GreedyLocalResult:
    """Assign services once using a local earliest-completion score.

    The procedure follows the joint DAG's topological order, enforces capacity
    and determined predecessor links at each step, and never backtracks. It is
    intentionally independent from the bounded constructive fallback.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    selected = np.full(instance.num_services, -1, dtype=np.int64)
    load = np.zeros_like(instance.device_capacity, dtype=np.float64)
    completion = np.zeros(instance.num_services, dtype=np.float64)
    pair_costs = build_dependency_pair_costs(instance)
    incoming: list[list[int]] = [[] for _ in range(instance.num_services)]
    for edge, target in enumerate(instance.dependency_index[1]):
        incoming[int(target)].append(edge)

    for assigned_count, raw_service in enumerate(instance.topological_order):
        service = int(raw_service)
        demand = instance.service_demand[service].astype(np.float64)
        options: list[tuple[float, float, int]] = []
        for raw_device in np.flatnonzero(instance.compatibility_mask[service]):
            device = int(raw_device)
            projected = load[device] + demand
            if (
                projected - instance.device_capacity[device].astype(np.float64)
                > tolerance
            ).any():
                continue

            arrivals: list[float] = []
            admissible = True
            for edge in incoming[service]:
                predecessor = int(instance.dependency_index[0, edge])
                predecessor_device = int(selected[predecessor])
                if predecessor_device < 0:
                    raise RuntimeError("Topological greedy order violated a dependency.")
                if not pair_costs.admissible[edge, predecessor_device, device]:
                    admissible = False
                    break
                arrivals.append(
                    completion[predecessor]
                    + float(
                        pair_costs.transmission_latency[
                            edge, predecessor_device, device
                        ]
                    )
                )
            if not admissible:
                continue

            estimated_completion = float(instance.processing_latency[service, device])
            if arrivals:
                estimated_completion += max(arrivals)
            utilization = projected / np.maximum(
                instance.device_capacity[device].astype(np.float64),
                tolerance,
            )
            options.append((estimated_completion, float(utilization.sum()), device))

        if not options:
            return GreedyLocalResult(
                placement=None,
                success=False,
                assigned_services=assigned_count,
                failed_service=service,
                verification=None,
            )

        estimated_completion, _, device = min(options)
        selected[service] = device
        load[device] += demand
        completion[service] = estimated_completion

    verification = verify_placement(instance, selected, tolerance=tolerance)
    return GreedyLocalResult(
        placement=selected.copy() if verification.feasible else None,
        success=verification.feasible,
        assigned_services=instance.num_services,
        failed_service=None,
        verification=verification,
    )


def _incident_edges(instance: DeploymentInstance) -> list[list[int]]:
    incident: list[list[int]] = [[] for _ in range(instance.num_services)]
    for edge, (source, target) in enumerate(instance.dependency_index.T):
        incident[int(source)].append(edge)
        incident[int(target)].append(edge)
    return incident


def _incoming_edges(instance: DeploymentInstance) -> list[list[int]]:
    incoming: list[list[int]] = [[] for _ in range(instance.num_services)]
    for edge, target in enumerate(instance.dependency_index[1]):
        incoming[int(target)].append(edge)
    return incoming


def _service_criticality(instance: DeploymentInstance) -> np.ndarray:
    """Static priority signal used only for deterministic heuristic ordering."""

    application_weight = instance.application_weight.astype(np.float64)
    membership_weight = instance.membership.astype(np.float64).T @ application_weight
    sink_weight = instance.sink_mask.astype(np.float64).T @ application_weight
    out_degree = np.bincount(
        instance.dependency_index[0],
        minlength=instance.num_services,
    ).astype(np.float64)
    in_degree = np.bincount(
        instance.dependency_index[1],
        minlength=instance.num_services,
    ).astype(np.float64)
    return membership_weight + 2.0 * sink_weight + 0.1 * (out_degree + in_degree)


def _partial_device_options(
    instance: DeploymentInstance,
    selected: np.ndarray,
    load: np.ndarray,
    completion: np.ndarray,
    service: int,
    *,
    pair_costs: Any,
    incident: list[list[int]],
    incoming: list[list[int]],
    scoring: str,
    tolerance: float,
) -> list[tuple[float, float, int]]:
    demand = instance.service_demand[service].astype(np.float64)
    options: list[tuple[float, float, int]] = []
    for raw_device in np.flatnonzero(instance.compatibility_mask[service]):
        device = int(raw_device)
        projected = load[device] + demand
        if (
            projected - instance.device_capacity[device].astype(np.float64)
            > tolerance
        ).any():
            continue

        link_sum = 0.0
        admissible = True
        for edge in incident[service]:
            source, target = instance.dependency_index[:, edge]
            other = int(target if int(source) == service else source)
            other_device = int(selected[other])
            if other_device < 0:
                continue
            source_device = device if int(source) == service else other_device
            target_device = device if int(target) == service else other_device
            if not pair_costs.admissible[edge, source_device, target_device]:
                admissible = False
                break
            link_sum += float(
                pair_costs.transmission_latency[edge, source_device, target_device]
            )
        if not admissible:
            continue

        arrivals = []
        for edge in incoming[service]:
            predecessor = int(instance.dependency_index[0, edge])
            predecessor_device = int(selected[predecessor])
            if predecessor_device < 0:
                continue
            arrivals.append(
                completion[predecessor]
                + float(pair_costs.transmission_latency[edge, predecessor_device, device])
            )
        estimated_completion = float(instance.processing_latency[service, device])
        if arrivals:
            estimated_completion += max(arrivals)

        utilization = projected / np.maximum(
            instance.device_capacity[device].astype(np.float64),
            tolerance,
        )
        utilization_sum = float(utilization.sum())
        if scoring == "latency":
            score = estimated_completion + 0.05 * link_sum + 0.01 * utilization_sum
        elif scoring == "balanced":
            score = (
                estimated_completion
                + 0.05 * link_sum
                + 0.15 * utilization_sum
                + 0.05 * float(np.max(utilization))
            )
        else:
            raise ValueError(f"Unsupported greedy scoring mode: {scoring!r}.")
        options.append((score, utilization_sum, device))
    options.sort()
    return options


def _select_greedy_service(
    instance: DeploymentInstance,
    selected: np.ndarray,
    load: np.ndarray,
    completion: np.ndarray,
    *,
    pair_costs: Any,
    incident: list[list[int]],
    incoming: list[list[int]],
    criticality: np.ndarray,
    strategy: str,
    scoring: str,
    topological_position: int,
    tolerance: float,
) -> tuple[int | None, list[tuple[float, float, int]], int | None]:
    if strategy == "topological":
        service = int(instance.topological_order[topological_position])
        if selected[service] >= 0:
            raise RuntimeError("Greedy topological position selected an assigned service.")
        options = _partial_device_options(
            instance,
            selected,
            load,
            completion,
            service,
            pair_costs=pair_costs,
            incident=incident,
            incoming=incoming,
            scoring=scoring,
            tolerance=tolerance,
        )
        return (service if options else None), options, service if not options else None

    best_key: tuple[float, ...] | None = None
    best_service: int | None = None
    best_options: list[tuple[float, float, int]] = []
    first_failed: int | None = None
    for raw_service in np.flatnonzero(selected < 0):
        service = int(raw_service)
        options = _partial_device_options(
            instance,
            selected,
            load,
            completion,
            service,
            pair_costs=pair_costs,
            incident=incident,
            incoming=incoming,
            scoring=scoring,
            tolerance=tolerance,
        )
        if not options:
            first_failed = service if first_failed is None else first_failed
            continue
        if strategy == "mrv":
            key = (len(options), -float(criticality[service]), service)
        elif strategy == "criticality":
            key = (-float(criticality[service]), len(options), service)
        else:
            raise ValueError(f"Unsupported greedy strategy: {strategy!r}.")
        if best_key is None or key < best_key:
            best_key = key
            best_service = service
            best_options = options
    return best_service, best_options, first_failed


def construct_latency_aware_heuristic_placement(
    instance: DeploymentInstance,
    *,
    strategy: str,
    scoring: str = "latency",
    tolerance: float = 1e-8,
) -> LatencyAwareHeuristicCandidate:
    """Build one deterministic placement with a named latency-aware rule."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if strategy not in {"topological", "mrv", "criticality"}:
        raise ValueError(f"Unsupported greedy strategy: {strategy!r}.")
    if scoring not in {"latency", "balanced"}:
        raise ValueError(f"Unsupported greedy scoring: {scoring!r}.")

    selected = np.full(instance.num_services, -1, dtype=np.int64)
    load = np.zeros_like(instance.device_capacity, dtype=np.float64)
    completion = np.zeros(instance.num_services, dtype=np.float64)
    pair_costs = build_dependency_pair_costs(instance)
    incident = _incident_edges(instance)
    incoming = _incoming_edges(instance)
    criticality = _service_criticality(instance)

    failed_service: int | None = None
    for assigned_count in range(instance.num_services):
        service, options, failed = _select_greedy_service(
            instance,
            selected,
            load,
            completion,
            pair_costs=pair_costs,
            incident=incident,
            incoming=incoming,
            criticality=criticality,
            strategy=strategy,
            scoring=scoring,
            topological_position=assigned_count,
            tolerance=tolerance,
        )
        if service is None:
            failed_service = failed
            return LatencyAwareHeuristicCandidate(
                placement=None,
                success=False,
                strategy=strategy,
                scoring=scoring,
                assigned_services=assigned_count,
                failed_service=failed_service,
                objective=None,
                verification=None,
            )

        _, _, device = options[0]
        selected[service] = device
        load[device] += instance.service_demand[service].astype(np.float64)
        arrivals = []
        for edge in incoming[service]:
            predecessor = int(instance.dependency_index[0, edge])
            predecessor_device = int(selected[predecessor])
            if predecessor_device >= 0:
                arrivals.append(
                    completion[predecessor]
                    + float(
                        pair_costs.transmission_latency[
                            edge, predecessor_device, device
                        ]
                    )
                )
        completion[service] = float(instance.processing_latency[service, device]) + (
            max(arrivals) if arrivals else 0.0
        )

    verification = verify_placement(instance, selected, tolerance=tolerance)
    objective = (
        evaluate_latency(instance, selected).objective
        if verification.feasible
        else None
    )
    return LatencyAwareHeuristicCandidate(
        placement=selected.copy() if verification.feasible else None,
        success=verification.feasible,
        strategy=strategy,
        scoring=scoring,
        assigned_services=instance.num_services,
        failed_service=None,
        objective=objective,
        verification=verification,
    )


def construct_latency_aware_heuristic_candidates(
    instance: DeploymentInstance,
    *,
    strategies: Iterable[str] = ("topological", "mrv", "criticality"),
    scoring_modes: Iterable[str] = ("latency", "balanced"),
    tolerance: float = 1e-8,
) -> LatencyAwareHeuristicResult:
    """Run a compact deterministic heuristic portfolio and select by exact latency."""

    candidates = tuple(
        construct_latency_aware_heuristic_placement(
            instance,
            strategy=strategy,
            scoring=scoring,
            tolerance=tolerance,
        )
        for strategy in strategies
        for scoring in scoring_modes
    )
    feasible = [candidate for candidate in candidates if candidate.success]
    if not feasible:
        return LatencyAwareHeuristicResult(
            placement=None,
            success=False,
            objective=None,
            selected_strategy=None,
            selected_scoring=None,
            candidates=candidates,
        )
    best = min(feasible, key=lambda candidate: float(candidate.objective))
    assert best.placement is not None and best.objective is not None
    return LatencyAwareHeuristicResult(
        placement=best.placement.copy(),
        success=True,
        objective=float(best.objective),
        selected_strategy=best.strategy,
        selected_scoring=best.scoring,
        candidates=candidates,
    )


def _empty_metrics(
    *,
    proposal_method: str,
    total_seconds: float,
    optimization_seconds: float,
) -> dict[str, Any]:
    return {
        "proposal_method": proposal_method,
        "num_raw_proposals": 0,
        "raw_feasible_count": 0,
        "raw_feasible_rate": None,
        "raw_capacity_violation_count": 0,
        "raw_capacity_violation_rate": None,
        "raw_link_violation_count": 0,
        "raw_link_violation_rate": None,
        "repair_attempts": 0,
        "repair_successes": 0,
        "repair_success_rate": 0.0,
        "total_repair_moves": 0,
        "fallback_invoked": False,
        "fallback_success": False,
        "fallback_search_nodes": 0,
        "final_success": False,
        "sampling_seconds": 0.0,
        "optimization_seconds": optimization_seconds,
        "verification_seconds": 0.0,
        "repair_seconds": 0.0,
        "fallback_seconds": 0.0,
        "exact_evaluation_seconds": 0.0,
        "selection_seconds": 0.0,
        "post_sampling_seconds": total_seconds,
        "total_seconds": total_seconds,
    }


def solve_greedy_local(instance: DeploymentInstance) -> SolveResult:
    """Run the independent no-backtracking local greedy baseline."""

    start = perf_counter()
    greedy = construct_greedy_local_placement(instance)
    construction_seconds = perf_counter() - start
    metrics = _empty_metrics(
        proposal_method="greedy_local",
        total_seconds=construction_seconds,
        optimization_seconds=construction_seconds,
    )
    metrics.update(
        {
            "num_raw_proposals": 1,
            "raw_feasible_count": int(greedy.success),
            "raw_feasible_rate": float(greedy.success),
            "greedy_assigned_services": greedy.assigned_services,
            "greedy_failed_service": greedy.failed_service,
            "final_success": greedy.success,
        }
    )
    if greedy.placement is None:
        return SolveResult(
            instance_id=instance.instance_id,
            placement=None,
            objective=None,
            source="failure",
            verified_candidates=(),
            metrics=metrics,
        )

    evaluation_start = perf_counter()
    objective = evaluate_latency(instance, greedy.placement).objective
    metrics["exact_evaluation_seconds"] = perf_counter() - evaluation_start
    metrics["total_seconds"] = perf_counter() - start
    candidate = VerifiedCandidate(
        placement=greedy.placement.copy(),
        objective=objective,
        source="greedy_local",
        proposal_index=None,
        repair_moves=0,
    )
    return SolveResult(
        instance_id=instance.instance_id,
        placement=greedy.placement.copy(),
        objective=objective,
        source="greedy_local",
        verified_candidates=(candidate,),
        metrics=metrics,
    )


def solve_latency_aware_heuristic(instance: DeploymentInstance) -> SolveResult:
    """Run the deterministic latency-aware heuristic baseline.

    The baseline is deterministic and non-learned: it builds a small set of
    feasible deployments using latency- and capacity-aware ordering rules,
    then selects the placement with the lowest exact end-to-end latency.
    """

    start = perf_counter()
    portfolio = construct_latency_aware_heuristic_candidates(instance)
    construction_seconds = perf_counter() - start
    metrics = _empty_metrics(
        proposal_method="latency_aware_heuristic",
        total_seconds=construction_seconds,
        optimization_seconds=construction_seconds,
    )
    feasible_count = sum(int(candidate.success) for candidate in portfolio.candidates)
    metrics.update(
        {
            "num_raw_proposals": len(portfolio.candidates),
            "raw_feasible_count": feasible_count,
            "raw_feasible_rate": (
                feasible_count / len(portfolio.candidates) if portfolio.candidates else None
            ),
            "heuristic_candidate_count": len(portfolio.candidates),
            "heuristic_successful_candidates": feasible_count,
            "heuristic_selected_strategy": portfolio.selected_strategy,
            "heuristic_selected_scoring": portfolio.selected_scoring,
            "heuristic_candidate_summary": [
                {
                    "strategy": candidate.strategy,
                    "scoring": candidate.scoring,
                    "success": candidate.success,
                    "assigned_services": candidate.assigned_services,
                    "failed_service": candidate.failed_service,
                    "objective": candidate.objective,
                }
                for candidate in portfolio.candidates
            ],
            "final_success": portfolio.success,
        }
    )
    if portfolio.placement is None or portfolio.objective is None:
        return SolveResult(
            instance_id=instance.instance_id,
            placement=None,
            objective=None,
            source="failure",
            verified_candidates=(),
            metrics=metrics,
        )

    metrics["exact_evaluation_seconds"] = perf_counter() - start - construction_seconds
    metrics["total_seconds"] = perf_counter() - start
    candidate = VerifiedCandidate(
        placement=portfolio.placement.copy(),
        objective=float(portfolio.objective),
        source="latency_aware_heuristic",
        proposal_index=None,
        repair_moves=0,
    )
    return SolveResult(
        instance_id=instance.instance_id,
        placement=portfolio.placement.copy(),
        objective=float(portfolio.objective),
        source="latency_aware_heuristic",
        verified_candidates=(candidate,),
        metrics=metrics,
    )


def _initial_local_search_candidates(
    instance: DeploymentInstance,
    config: LocalSearchConfig,
) -> list[tuple[str, np.ndarray, float]]:
    candidates: list[tuple[str, np.ndarray, float]] = []
    heuristic = construct_latency_aware_heuristic_candidates(
        instance,
        tolerance=config.tolerance,
    )
    if heuristic.placement is not None and heuristic.objective is not None:
        candidates.append(
            (
                "latency_aware_heuristic",
                heuristic.placement,
                float(heuristic.objective),
            )
        )

    fallback = construct_feasible_placement(
        instance,
        config=ConstructiveFallbackConfig(
            max_search_nodes=config.fallback_max_search_nodes,
            tolerance=config.tolerance,
        ),
    )
    if fallback.placement is not None and fallback.success:
        objective = evaluate_latency(instance, fallback.placement).objective
        candidates.append(("fallback", fallback.placement, objective))
    return candidates


def run_local_search(
    instance: DeploymentInstance,
    *,
    config: LocalSearchConfig | None = None,
) -> LocalSearchResult:
    """Strictly improve a feasible placement by single-service relocation."""

    settings = config or LocalSearchConfig()
    settings.validate()
    initial = _initial_local_search_candidates(instance, settings)
    if not initial:
        return LocalSearchResult(
            placement=None,
            success=False,
            objective=None,
            initial_source=None,
            initial_objective=None,
            accepted_moves=0,
            evaluated_moves=0,
            passes_completed=0,
            exhausted_budget=False,
        )

    source, placement, objective = min(initial, key=lambda row: row[2])
    best = placement.copy()
    best_objective = float(objective)
    initial_objective = float(objective)
    criticality = _service_criticality(instance)
    service_order = sorted(
        range(instance.num_services),
        key=lambda service: (-float(criticality[service]), service),
    )
    evaluated = 0
    accepted = 0
    passes_completed = 0
    exhausted = False

    for _ in range(settings.max_passes):
        improved = False
        passes_completed += 1
        for service in service_order:
            current_device = int(best[service])
            device_order = sorted(
                (
                    int(device)
                    for device in np.flatnonzero(instance.compatibility_mask[service])
                    if int(device) != current_device
                ),
                key=lambda device: (
                    float(instance.processing_latency[service, device]),
                    device,
                ),
            )
            for device in device_order:
                if evaluated >= settings.max_evaluations:
                    exhausted = True
                    break
                evaluated += 1
                trial = best.copy()
                trial[service] = device
                verification = verify_placement(
                    instance, trial, tolerance=settings.tolerance
                )
                if not verification.feasible:
                    continue
                trial_objective = evaluate_latency(instance, trial).objective
                if trial_objective + settings.min_improvement < best_objective:
                    best = verification.placement.copy()
                    best_objective = float(trial_objective)
                    accepted += 1
                    improved = True
                    break
            if exhausted:
                break
        if exhausted or not improved:
            break

    return LocalSearchResult(
        placement=best.copy(),
        success=True,
        objective=best_objective,
        initial_source=source,
        initial_objective=initial_objective,
        accepted_moves=accepted,
        evaluated_moves=evaluated,
        passes_completed=passes_completed,
        exhausted_budget=exhausted,
    )


def solve_local_search(
    instance: DeploymentInstance,
    *,
    config: LocalSearchConfig | None = None,
) -> SolveResult:
    """Run Local Search as a non-learned optimization baseline."""

    settings = config or LocalSearchConfig()
    settings.validate()
    start = perf_counter()
    result = run_local_search(instance, config=settings)
    total_seconds = perf_counter() - start
    metrics = _empty_metrics(
        proposal_method="local_search",
        total_seconds=total_seconds,
        optimization_seconds=total_seconds,
    )
    improvement = (
        None
        if result.initial_objective is None
        or result.objective is None
        or result.initial_objective == 0
        else (result.initial_objective - result.objective) / result.initial_objective
    )
    metrics.update(
        {
            "final_success": result.success,
            "local_search_initial_source": result.initial_source,
            "local_search_initial_objective": result.initial_objective,
            "local_search_accepted_moves": result.accepted_moves,
            "local_search_evaluated_moves": result.evaluated_moves,
            "local_search_passes_completed": result.passes_completed,
            "local_search_exhausted_budget": result.exhausted_budget,
            "local_search_relative_improvement": improvement,
        }
    )
    if result.placement is None or result.objective is None:
        return SolveResult(
            instance_id=instance.instance_id,
            placement=None,
            objective=None,
            source="failure",
            verified_candidates=(),
            metrics=metrics,
        )
    candidate = VerifiedCandidate(
        placement=result.placement.copy(),
        objective=float(result.objective),
        source="local_search",
        proposal_index=None,
        repair_moves=result.accepted_moves,
    )
    return SolveResult(
        instance_id=instance.instance_id,
        placement=result.placement.copy(),
        objective=float(result.objective),
        source="local_search",
        verified_candidates=(candidate,),
        metrics=metrics,
    )


def solve_milp_time_limit(
    instance: DeploymentInstance,
    *,
    time_limit_seconds: float,
    seed: int,
    threads: int = 1,
) -> SolveResult:
    """Run the exact MILP under a real solver-enforced time limit."""

    solver_seed = int(seed) % 2_000_000_000
    start = perf_counter()
    milp = solve_milp(
        instance,
        MilpConfig(
            time_limit_seconds=time_limit_seconds,
            mip_gap=0.0,
            threads=threads,
            seed=solver_seed,
            output_flag=False,
        ),
    )
    total_seconds = perf_counter() - start
    metrics = _empty_metrics(
        proposal_method="milp_time_limit",
        total_seconds=total_seconds,
        optimization_seconds=milp.runtime_seconds,
    )
    metrics.update(
        {
            "milp": {
                **asdict(milp),
                "placement": (
                    None if milp.placement is None else milp.placement.tolist()
                ),
            },
            "milp_status": milp.status_name,
            "milp_optimal": milp.optimal,
            "milp_objective_bound": milp.objective_bound,
            "milp_gap": milp.mip_gap,
            "milp_solver_runtime_seconds": milp.runtime_seconds,
            "milp_threads": threads,
            "milp_experiment_seed": int(seed),
            "milp_solver_seed": solver_seed,
            "final_success": milp.placement is not None,
        }
    )
    if milp.placement is None or milp.exact_objective is None:
        return SolveResult(
            instance_id=instance.instance_id,
            placement=None,
            objective=None,
            source="failure",
            verified_candidates=(),
            metrics=metrics,
        )

    candidate = VerifiedCandidate(
        placement=milp.placement.copy(),
        objective=milp.exact_objective,
        source="milp_incumbent",
        proposal_index=None,
        repair_moves=0,
    )
    return SolveResult(
        instance_id=instance.instance_id,
        placement=milp.placement.copy(),
        objective=milp.exact_objective,
        source="milp_incumbent",
        verified_candidates=(candidate,),
        metrics=metrics,
    )
