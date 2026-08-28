"""Verified near-optimal solution pools for energy-weighted training."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gdm_factor_diffusion.data.schema import DeploymentInstance

from .milp import (
    MilpConfig,
    _import_gurobi,
    _status_name,
    add_no_good_cut,
    build_equivalent_milp,
    extract_incumbent,
)

SOLUTION_POOL_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class SolutionPoolConfig:
    target_size: int = 32
    beta: float = 5.0
    total_time_limit_seconds: float = 300.0
    mip_gap: float = 0.0
    threads: int | None = 1
    seed: int = 0
    output_flag: bool = False
    objective_tolerance: float = 1e-5
    energy_epsilon: float = 1e-12


@dataclass(slots=True)
class SolutionPool:
    instance_id: str
    placements: np.ndarray
    latencies: np.ndarray
    normalized_energy: np.ndarray
    sampling_probability: np.ndarray
    verified: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.instance_id = str(self.instance_id)
        self.placements = np.ascontiguousarray(self.placements, dtype=np.int64)
        self.latencies = np.ascontiguousarray(self.latencies, dtype=np.float64)
        self.normalized_energy = np.ascontiguousarray(
            self.normalized_energy, dtype=np.float64
        )
        self.sampling_probability = np.ascontiguousarray(
            self.sampling_probability, dtype=np.float64
        )
        self.verified = np.ascontiguousarray(self.verified, dtype=np.bool_)
        self.metadata = dict(self.metadata)
        self.validate()

    @property
    def size(self) -> int:
        return int(self.placements.shape[0])

    def validate(self) -> None:
        if not self.instance_id.strip():
            raise ValueError("Solution-pool instance_id must be nonempty.")
        if self.placements.ndim != 2:
            raise ValueError("Solution-pool placements must be [S, M].")
        size = self.size
        for name, array in (
            ("latencies", self.latencies),
            ("normalized_energy", self.normalized_energy),
            ("sampling_probability", self.sampling_probability),
            ("verified", self.verified),
        ):
            if array.shape != (size,):
                raise ValueError(f"{name} must have shape [S].")
        if size < 1:
            raise ValueError("A solution pool must contain at least one placement.")
        if np.unique(self.placements, axis=0).shape[0] != size:
            raise ValueError("Solution-pool placements must be unique.")
        if not np.isfinite(self.latencies).all() or (self.latencies < 0).any():
            raise ValueError("Solution-pool latencies must be finite and nonnegative.")
        if not np.isfinite(self.normalized_energy).all():
            raise ValueError("Normalized energies must be finite.")
        if (self.normalized_energy < 0).any() or (self.normalized_energy > 1).any():
            raise ValueError("Normalized energies must lie in [0, 1].")
        if not np.isfinite(self.sampling_probability).all():
            raise ValueError("Sampling probabilities must be finite.")
        if (self.sampling_probability < 0).any():
            raise ValueError("Sampling probabilities must be nonnegative.")
        if not np.isclose(self.sampling_probability.sum(), 1.0, atol=1e-10):
            raise ValueError("Sampling probabilities must sum to one.")
        if not self.verified.all():
            raise ValueError("Every saved solution-pool placement must be verified.")
        if self.metadata.get("schema_version") != SOLUTION_POOL_SCHEMA_VERSION:
            raise ValueError(
                "Solution-pool metadata has an unsupported schema version."
            )
        json.dumps(self.metadata, sort_keys=True)


def compute_energy_distribution(
    latencies: np.ndarray,
    *,
    beta: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    if beta < 0:
        raise ValueError("beta must be nonnegative.")
    if epsilon <= 0:
        raise ValueError("energy_epsilon must be positive.")
    minimum = float(latencies.min())
    maximum = float(latencies.max())
    normalized = (latencies - minimum) / (maximum - minimum + epsilon)
    logits = -float(beta) * normalized
    logits -= logits.max()
    probability = np.exp(logits)
    probability /= probability.sum()
    return normalized, probability


def build_solution_pool(
    instance: DeploymentInstance,
    config: SolutionPoolConfig | None = None,
) -> SolutionPool:
    """Enumerate distinct low-latency placements with repeated no-good cuts."""

    config = config or SolutionPoolConfig()
    if config.target_size < 1:
        raise ValueError("target_size must be positive.")
    if config.total_time_limit_seconds <= 0:
        raise ValueError("total_time_limit_seconds must be positive.")

    milp_config = MilpConfig(
        time_limit_seconds=config.total_time_limit_seconds,
        mip_gap=config.mip_gap,
        threads=config.threads,
        seed=config.seed,
        output_flag=config.output_flag,
        objective_tolerance=config.objective_tolerance,
    )
    artifacts = build_equivalent_milp(instance, milp_config)
    _, GRB = _import_gurobi()
    start = time.perf_counter()
    placements: list[np.ndarray] = []
    latencies: list[float] = []
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    termination_reason = "target_size_reached"

    while len(placements) < config.target_size:
        remaining = config.total_time_limit_seconds - (time.perf_counter() - start)
        if remaining <= 0:
            termination_reason = "total_time_limit"
            break
        artifacts.model.Params.TimeLimit = float(remaining)
        artifacts.model.optimize()
        status = int(artifacts.model.Status)
        if artifacts.model.SolCount < 1:
            termination_reason = _status_name(status, GRB).lower()
            break

        incumbent = extract_incumbent(
            instance,
            artifacts,
            objective_tolerance=config.objective_tolerance,
        )
        key = tuple(int(device) for device in incumbent.placement)
        if key in seen:
            raise RuntimeError("No-good-cut enumeration returned a duplicate placement.")
        seen.add(key)
        placements.append(incumbent.placement.copy())
        latencies.append(incumbent.exact_objective)
        records.append(
            {
                "rank_generated": len(placements) - 1,
                "status": _status_name(status, GRB),
                "optimal_under_exclusions": status == GRB.OPTIMAL,
                "solver_objective": incumbent.solver_objective,
                "exact_objective": incumbent.exact_objective,
                "objective_error": incumbent.objective_error,
                "mip_gap": float(artifacts.model.MIPGap),
                "runtime_seconds": float(artifacts.model.Runtime),
            }
        )
        add_no_good_cut(
            artifacts,
            incumbent.placement,
            name=f"exclude[{len(placements) - 1}]",
        )

    if not placements:
        raise RuntimeError(
            f"No feasible solution was collected for instance {instance.instance_id!r}."
        )

    placement_array = np.asarray(placements, dtype=np.int64)
    latency_array = np.asarray(latencies, dtype=np.float64)
    order = np.lexsort(tuple(placement_array[:, index] for index in reversed(
        range(instance.num_services)
    )) + (latency_array,))
    placement_array = placement_array[order]
    latency_array = latency_array[order]
    records = [records[int(index)] for index in order]
    for rank, record in enumerate(records):
        record["rank_sorted"] = rank
    normalized, probability = compute_energy_distribution(
        latency_array,
        beta=config.beta,
        epsilon=config.energy_epsilon,
    )
    elapsed = time.perf_counter() - start
    metadata = {
        "schema_version": SOLUTION_POOL_SCHEMA_VERSION,
        "generator": "repeated_milp_no_good_cuts",
        "requested_size": config.target_size,
        "actual_size": int(placement_array.shape[0]),
        "beta": config.beta,
        "energy_epsilon": config.energy_epsilon,
        "minimum_latency": float(latency_array.min()),
        "maximum_latency": float(latency_array.max()),
        "elapsed_seconds": elapsed,
        "termination_reason": termination_reason,
        "config": asdict(config),
        "solve_records": records,
    }
    return SolutionPool(
        instance_id=instance.instance_id,
        placements=placement_array,
        latencies=latency_array,
        normalized_energy=normalized,
        sampling_probability=probability,
        verified=np.ones(placement_array.shape[0], dtype=np.bool_),
        metadata=metadata,
    )


def save_solution_pool(pool: SolutionPool, path: str | Path) -> Path:
    """Atomically save a solution pool without pickle."""

    pool.validate()
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("Solution pools must use the .npz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            placements=pool.placements,
            latencies=pool.latencies,
            normalized_energy=pool.normalized_energy,
            sampling_probability=pool.sampling_probability,
            verified=pool.verified,
            _instance_id=np.asarray(pool.instance_id),
            _metadata_json=np.asarray(
                json.dumps(pool.metadata, sort_keys=True, separators=(",", ":"))
            ),
        )
    os.replace(temporary, destination)
    return destination


def load_solution_pool(path: str | Path) -> SolutionPool:
    """Load and validate a solution pool without pickle."""

    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "placements",
            "latencies",
            "normalized_energy",
            "sampling_probability",
            "verified",
            "_instance_id",
            "_metadata_json",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"Solution-pool file is missing fields: {sorted(missing)}")
        return SolutionPool(
            instance_id=str(payload["_instance_id"].item()),
            placements=payload["placements"],
            latencies=payload["latencies"],
            normalized_energy=payload["normalized_energy"],
            sampling_probability=payload["sampling_probability"],
            verified=payload["verified"],
            metadata=json.loads(str(payload["_metadata_json"].item())),
        )
