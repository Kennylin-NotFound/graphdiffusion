"""Equivalent MILP for the joint-DAG end-to-end latency objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gdm_factor_diffusion.data.schema import DeploymentInstance

from .latency_evaluator import evaluate_latency
from .pair_costs import build_dependency_pair_costs
from .placement_verifier import verify_placement


@dataclass(frozen=True, slots=True)
class MilpConfig:
    """Deterministic Gurobi settings for one placement solve."""

    time_limit_seconds: float | None = None
    mip_gap: float = 0.0
    threads: int | None = 1
    seed: int = 0
    output_flag: bool = False
    objective_tolerance: float = 1e-5


@dataclass(slots=True)
class MilpArtifacts:
    """Gurobi model and variables needed for solving and no-good cuts."""

    model: Any
    x: dict[tuple[int, int], Any]
    y: dict[tuple[int, int, int], Any]
    completion_time: dict[int, Any]
    application_latency: dict[int, Any]


@dataclass(frozen=True, slots=True)
class MilpIncumbent:
    placement: np.ndarray
    solver_objective: float
    exact_objective: float
    objective_error: float


@dataclass(frozen=True, slots=True)
class MilpSolveResult:
    instance_id: str
    status: int
    status_name: str
    optimal: bool
    placement: np.ndarray | None
    solver_objective: float | None
    exact_objective: float | None
    objective_error: float | None
    objective_bound: float | None
    mip_gap: float | None
    runtime_seconds: float
    num_variables: int
    num_constraints: int


def _import_gurobi() -> tuple[Any, Any]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise RuntimeError(
            "gurobipy is required for Phase 1C MILP generation."
        ) from error
    return gp, GRB


def _validate_config(config: MilpConfig) -> None:
    if config.time_limit_seconds is not None and config.time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive when specified.")
    if config.mip_gap < 0:
        raise ValueError("mip_gap must be nonnegative.")
    if config.threads is not None and config.threads < 1:
        raise ValueError("threads must be positive when specified.")
    if config.objective_tolerance <= 0:
        raise ValueError("objective_tolerance must be positive.")


def _configure_model(model: Any, config: MilpConfig) -> None:
    model.Params.OutputFlag = int(config.output_flag)
    model.Params.MIPGap = float(config.mip_gap)
    model.Params.Seed = int(config.seed)
    if config.threads is not None:
        model.Params.Threads = int(config.threads)
    if config.time_limit_seconds is not None:
        model.Params.TimeLimit = float(config.time_limit_seconds)


def build_equivalent_milp(
    instance: DeploymentInstance,
    config: MilpConfig | None = None,
    *,
    environment: Any | None = None,
) -> MilpArtifacts:
    """Build the exact discrete placement MILP used for offline labels."""

    config = config or MilpConfig()
    _validate_config(config)
    gp, GRB = _import_gurobi()
    model_name = f"placement_{instance.instance_id}".replace("-", "_")
    model = gp.Model(model_name, env=environment) if environment else gp.Model(model_name)
    _configure_model(model, config)

    x: dict[tuple[int, int], Any] = {}
    for service in range(instance.num_services):
        for device in np.flatnonzero(instance.compatibility_mask[service]):
            device = int(device)
            x[service, device] = model.addVar(
                vtype=GRB.BINARY,
                name=f"x[{service},{device}]",
            )

    for service in range(instance.num_services):
        candidates = np.flatnonzero(instance.compatibility_mask[service])
        model.addConstr(
            gp.quicksum(x[service, int(device)] for device in candidates) == 1,
            name=f"assign[{service}]",
        )

    for device in range(instance.num_devices):
        for resource in range(instance.num_resources):
            model.addConstr(
                gp.quicksum(
                    float(instance.service_demand[service, resource])
                    * x[service, device]
                    for service in range(instance.num_services)
                    if (service, device) in x
                )
                <= float(instance.device_capacity[device, resource]),
                name=f"capacity[{device},{resource}]",
            )

    pair_costs = build_dependency_pair_costs(instance)
    y: dict[tuple[int, int, int], Any] = {}
    for edge in range(instance.num_dependencies):
        source = int(instance.dependency_index[0, edge])
        target = int(instance.dependency_index[1, edge])
        admissible_pairs = np.argwhere(pair_costs.admissible[edge])
        for source_device, target_device in admissible_pairs:
            key = edge, int(source_device), int(target_device)
            y[key] = model.addVar(vtype=GRB.BINARY, name=f"y[{edge},{key[1]},{key[2]}]")

        edge_variables = [
            variable for (candidate_edge, _, _), variable in y.items()
            if candidate_edge == edge
        ]
        model.addConstr(gp.quicksum(edge_variables) == 1, name=f"pair[{edge}]")

        for source_device in np.flatnonzero(instance.compatibility_mask[source]):
            source_device = int(source_device)
            model.addConstr(
                gp.quicksum(
                    variable
                    for (candidate_edge, i, _), variable in y.items()
                    if candidate_edge == edge and i == source_device
                )
                == x[source, source_device],
                name=f"pair_source[{edge},{source_device}]",
            )
        for target_device in np.flatnonzero(instance.compatibility_mask[target]):
            target_device = int(target_device)
            model.addConstr(
                gp.quicksum(
                    variable
                    for (candidate_edge, _, j), variable in y.items()
                    if candidate_edge == edge and j == target_device
                )
                == x[target, target_device],
                name=f"pair_target[{edge},{target_device}]",
            )

    completion_time = {
        service: model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"T[{service}]")
        for service in range(instance.num_services)
    }
    application_latency = {
        application: model.addVar(
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name=f"L[{application}]",
        )
        for application in range(instance.num_applications)
    }

    processing = {
        service: gp.quicksum(
            float(instance.processing_latency[service, device]) * variable
            for (candidate_service, device), variable in x.items()
            if candidate_service == service
        )
        for service in range(instance.num_services)
    }
    transmission = {
        edge: gp.quicksum(
            float(pair_costs.transmission_latency[edge, source_device, target_device])
            * variable
            for (candidate_edge, source_device, target_device), variable in y.items()
            if candidate_edge == edge
        )
        for edge in range(instance.num_dependencies)
    }

    incoming_count = np.zeros(instance.num_services, dtype=np.int64)
    for edge, (source, target) in enumerate(instance.dependency_index.T):
        source = int(source)
        target = int(target)
        incoming_count[target] += 1
        model.addConstr(
            completion_time[target]
            >= completion_time[source] + transmission[edge] + processing[target],
            name=f"completion_edge[{edge}]",
        )
    for service in np.flatnonzero(incoming_count == 0):
        service = int(service)
        model.addConstr(
            completion_time[service] >= processing[service],
            name=f"completion_source[{service}]",
        )

    for application in range(instance.num_applications):
        for sink in np.flatnonzero(instance.sink_mask[application]):
            sink = int(sink)
            model.addConstr(
                application_latency[application] >= completion_time[sink],
                name=f"application_sink[{application},{sink}]",
            )

    model.setObjective(
        gp.quicksum(
            float(instance.application_weight[application])
            * application_latency[application]
            for application in range(instance.num_applications)
        ),
        GRB.MINIMIZE,
    )
    model.update()
    return MilpArtifacts(
        model=model,
        x=x,
        y=y,
        completion_time=completion_time,
        application_latency=application_latency,
    )


def extract_incumbent(
    instance: DeploymentInstance,
    artifacts: MilpArtifacts,
    *,
    objective_tolerance: float = 1e-5,
) -> MilpIncumbent:
    """Extract, verify, and exactly re-evaluate the current Gurobi incumbent."""

    if artifacts.model.SolCount < 1:
        raise ValueError("The MILP model has no incumbent solution.")
    placement = np.empty(instance.num_services, dtype=np.int64)
    for service in range(instance.num_services):
        candidates = [
            (device, variable.X)
            for (candidate_service, device), variable in artifacts.x.items()
            if candidate_service == service
        ]
        placement[service] = max(candidates, key=lambda item: item[1])[0]

    verification = verify_placement(instance, placement)
    if not verification.feasible:
        raise RuntimeError(
            "MILP incumbent failed the shared verifier: "
            f"{verification.to_dict()}"
        )
    exact_objective = evaluate_latency(instance, placement).objective
    solver_objective = float(artifacts.model.ObjVal)
    objective_error = abs(solver_objective - exact_objective)
    if objective_error > objective_tolerance:
        raise RuntimeError(
            "MILP and exact evaluator objectives disagree: "
            f"solver={solver_objective:.12g}, exact={exact_objective:.12g}, "
            f"error={objective_error:.3g}."
        )
    return MilpIncumbent(
        placement=placement,
        solver_objective=solver_objective,
        exact_objective=exact_objective,
        objective_error=objective_error,
    )


def add_no_good_cut(
    artifacts: MilpArtifacts,
    placement: np.ndarray,
    *,
    name: str | None = None,
) -> Any:
    """Exclude exactly one categorical placement from subsequent solves."""

    gp, _ = _import_gurobi()
    selected = np.asarray(placement, dtype=np.int64)
    if selected.shape != (len(artifacts.completion_time),):
        raise ValueError("No-good placement has the wrong number of services.")
    return artifacts.model.addConstr(
        gp.quicksum(
            artifacts.x[service, int(device)]
            for service, device in enumerate(selected)
        )
        <= selected.size - 1,
        name=name,
    )


def _status_name(status: int, grb: Any) -> str:
    names = {
        grb.LOADED: "LOADED",
        grb.OPTIMAL: "OPTIMAL",
        grb.INFEASIBLE: "INFEASIBLE",
        grb.INF_OR_UNBD: "INF_OR_UNBD",
        grb.UNBOUNDED: "UNBOUNDED",
        grb.CUTOFF: "CUTOFF",
        grb.ITERATION_LIMIT: "ITERATION_LIMIT",
        grb.NODE_LIMIT: "NODE_LIMIT",
        grb.TIME_LIMIT: "TIME_LIMIT",
        grb.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        grb.INTERRUPTED: "INTERRUPTED",
        grb.NUMERIC: "NUMERIC",
        grb.SUBOPTIMAL: "SUBOPTIMAL",
        grb.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return names.get(status, f"STATUS_{status}")


def solve_milp(
    instance: DeploymentInstance,
    config: MilpConfig | None = None,
) -> MilpSolveResult:
    """Solve one equivalent MILP and cross-check its incumbent."""

    config = config or MilpConfig()
    artifacts = build_equivalent_milp(instance, config)
    artifacts.model.optimize()
    _, GRB = _import_gurobi()
    status = int(artifacts.model.Status)
    incumbent = (
        extract_incumbent(
            instance,
            artifacts,
            objective_tolerance=config.objective_tolerance,
        )
        if artifacts.model.SolCount > 0
        else None
    )
    return MilpSolveResult(
        instance_id=instance.instance_id,
        status=status,
        status_name=_status_name(status, GRB),
        optimal=status == GRB.OPTIMAL,
        placement=None if incumbent is None else incumbent.placement,
        solver_objective=None if incumbent is None else incumbent.solver_objective,
        exact_objective=None if incumbent is None else incumbent.exact_objective,
        objective_error=None if incumbent is None else incumbent.objective_error,
        objective_bound=(
            float(artifacts.model.ObjBound) if artifacts.model.IsMIP else None
        ),
        mip_gap=(
            float(artifacts.model.MIPGap) if artifacts.model.SolCount > 0 else None
        ),
        runtime_seconds=float(artifacts.model.Runtime),
        num_variables=int(artifacts.model.NumVars),
        num_constraints=int(artifacts.model.NumConstrs),
    )
