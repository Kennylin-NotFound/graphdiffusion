"""Prepare a realistic-profile synthetic dataset.

The script keeps the existing optimization model and schema intact. It first
generates graph-ready instances from a normal dataset config, then transforms
workloads, device rates/capacities, and effective direct-link rates according
to the realistic-profile design note.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed
from gdm_factor_diffusion.data import (
    audit_graph_readiness,
    generate_dataset,
    load_manifest,
    load_manifest_instance,
    save_instance,
)
from gdm_factor_diffusion.solver import evaluate_latency, verify_placement


TRANSFORM_VERSION = "realistic_profile_v1"

SERVICE_MULTIPLIERS: dict[int, dict[str, tuple[float, float]]] = {
    0: {"compute": (0.8, 1.2), "output": (1.2, 1.8), "resource": (0.8, 1.1)},
    1: {"compute": (0.8, 1.2), "output": (0.8, 1.3), "resource": (0.8, 1.1)},
    2: {"compute": (1.1, 1.6), "output": (0.5, 0.9), "resource": (1.1, 1.4)},
    3: {"compute": (0.9, 1.3), "output": (0.5, 0.8), "resource": (0.9, 1.2)},
    4: {"compute": (0.9, 1.3), "output": (0.4, 0.8), "resource": (0.9, 1.2)},
    5: {"compute": (1.2, 1.8), "output": (0.7, 1.1), "resource": (1.2, 1.5)},
    6: {"compute": (1.1, 1.6), "output": (0.6, 1.0), "resource": (1.1, 1.4)},
    7: {"compute": (0.9, 1.3), "output": (0.9, 1.5), "resource": (0.9, 1.2)},
    8: {"compute": (0.7, 1.0), "output": (0.3, 0.6), "resource": (0.7, 1.0)},
    9: {"compute": (0.6, 0.9), "output": (0.2, 0.4), "resource": (0.6, 0.9)},
}

DEVICE_MULTIPLIERS: dict[int, dict[str, tuple[float, float]]] = {
    0: {"frequency": (0.75, 1.15), "capacity": (0.80, 1.10)},
    1: {"frequency": (0.90, 1.30), "capacity": (0.90, 1.25)},
    2: {"frequency": (1.00, 1.50), "capacity": (1.00, 1.40)},
}

LINK_RATE_RANGES: dict[tuple[int, int], tuple[float, float]] = {
    (0, 0): (8.0, 25.0),
    (0, 1): (25.0, 80.0),
    (0, 2): (40.0, 120.0),
    (1, 1): (60.0, 160.0),
    (1, 2): (100.0, 240.0),
    (2, 2): (180.0, 400.0),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform(rng: np.random.Generator, interval: tuple[float, float]) -> float:
    return float(rng.uniform(float(interval[0]), float(interval[1])))


def _update_service_quantities(
    instance: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    processing = instance.service_features[:, 0].astype(np.float64).copy()
    output = instance.service_features[:, 1].astype(np.float64).copy()
    demand = instance.service_demand.astype(np.float64).copy()
    for service, type_id in enumerate(instance.service_type_id):
        spec = SERVICE_MULTIPLIERS[int(type_id)]
        processing[service] *= _uniform(rng, spec["compute"])
        output[service] *= _uniform(rng, spec["output"])
        demand[service] *= rng.uniform(*spec["resource"], size=demand.shape[1])
    return processing.astype(np.float32), output.astype(np.float32), demand.astype(np.float32)


def _update_device_quantities(
    instance: Any,
    service_demand: np.ndarray,
    witness: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = instance.device_features[:, 0].astype(np.float64).copy()
    capacity = instance.device_capacity.astype(np.float64).copy()
    for device, type_id in enumerate(instance.device_type_id):
        spec = DEVICE_MULTIPLIERS[int(type_id)]
        frequency[device] *= _uniform(rng, spec["frequency"])
        capacity[device] *= rng.uniform(*spec["capacity"], size=capacity.shape[1])

    witness_load = np.zeros_like(capacity)
    for service, device in enumerate(witness):
        witness_load[int(device)] += service_demand[service]
    slack = float(instance.metadata.get("capacity_slack", 0.20))
    capacity = np.maximum(capacity, witness_load * (1.0 + slack))
    capacity = np.maximum(capacity, witness_load + 1e-3)
    return frequency.astype(np.float32), capacity.astype(np.float32)


def _update_link_rates(instance: Any, rng: np.random.Generator) -> np.ndarray:
    link_rate = np.zeros_like(instance.link_rate, dtype=np.float32)
    connectivity = instance.connectivity
    for left in range(instance.num_devices):
        for right in range(left + 1, instance.num_devices):
            if not connectivity[left, right]:
                continue
            pair = tuple(sorted((int(instance.device_type_id[left]), int(instance.device_type_id[right]))))
            rate = _uniform(rng, LINK_RATE_RANGES[pair])
            link_rate[left, right] = link_rate[right, left] = rate
    return link_rate


def _transform_instance(instance: Any, witness: np.ndarray, seed: int) -> Any:
    rng = np.random.default_rng(seed)
    processing, output, service_demand = _update_service_quantities(instance, rng)
    frequency, capacity = _update_device_quantities(instance, service_demand, witness, rng)
    link_rate = _update_link_rates(instance, rng)

    processing_latency = np.zeros_like(instance.processing_latency, dtype=np.float32)
    all_latency = processing[:, None] / frequency[None, :]
    processing_latency[instance.compatibility_mask] = all_latency[instance.compatibility_mask]

    service_features = instance.service_features.copy()
    service_features[:, 0] = processing
    service_features[:, 1] = output
    service_features[:, 2 : 2 + service_demand.shape[1]] = service_demand

    degree = instance.connectivity.sum(axis=1).astype(np.float32)
    nonzero_rate_count = np.maximum(degree, 1.0)
    mean_link_rate = link_rate.sum(axis=1) / nonzero_rate_count
    device_features = instance.device_features.copy()
    device_features[:, 0] = frequency
    device_features[:, 1 : 1 + capacity.shape[1]] = capacity
    device_features[:, 1 + capacity.shape[1]] = degree
    device_features[:, 2 + capacity.shape[1]] = mean_link_rate

    source = instance.dependency_index[0]
    dependency_data_volume = output[source].astype(np.float32)
    metadata = dict(instance.metadata)
    metadata.update(
        {
            "profile_name": "realistic_edge_video",
            "profile_transform_version": TRANSFORM_VERSION,
            "profile_seed": int(seed),
            "profile_description": (
                "Scenario-specific synthetic profile for edge video analytics; "
                "direct-link/no-contention model is unchanged."
            ),
            "realistic_profile_units": "normalized_profiled_quantities",
        }
    )
    return type(instance)(
        instance_id=instance.instance_id,
        service_type_id=instance.service_type_id,
        service_features=service_features.astype(np.float32),
        service_demand=service_demand.astype(np.float32),
        processing_latency=processing_latency.astype(np.float32),
        compatibility_mask=instance.compatibility_mask,
        device_type_id=instance.device_type_id,
        device_features=device_features.astype(np.float32),
        device_capacity=capacity.astype(np.float32),
        connectivity=instance.connectivity,
        link_rate=link_rate.astype(np.float32),
        dependency_index=instance.dependency_index,
        dependency_data_volume=dependency_data_volume,
        application_weight=instance.application_weight,
        application_type_id=instance.application_type_id,
        membership=instance.membership,
        application_dependency_mask=instance.application_dependency_mask,
        sink_mask=instance.sink_mask,
        topological_order=instance.topological_order,
        metadata=metadata,
    )


def prepare(config_path: Path, output_root: Path, *, force: bool) -> dict[str, Any]:
    config = load_config(config_path)
    if not output_root.exists() or not (output_root / "manifest.json").is_file():
        generate_dataset(config, output_root=output_root)

    if (output_root / "solution_pool_manifest.json").exists() and not force:
        raise RuntimeError(
            "Refusing to transform a dataset that already has solution pools. "
            "Use a fresh output root or --force only if you intentionally want to "
            "invalidate old labels."
        )

    manifest = load_manifest(output_root / "manifest.json")
    already_done = all(
        load_manifest_instance(output_root, entry).metadata.get("profile_transform_version")
        == TRANSFORM_VERSION
        for entry in manifest["instances"][: min(3, len(manifest["instances"]))]
    )
    if already_done and not force:
        return {
            "dataset_root": str(output_root),
            "instances": manifest["instance_count"],
            "status": "already_transformed",
        }

    transformed = []
    for entry in manifest["instances"]:
        instance = load_manifest_instance(output_root, entry)
        witness = np.asarray(entry["witness_placement"], dtype=np.int64)
        seed = derive_seed(
            int(manifest["base_seed"]),
            f"{TRANSFORM_VERSION}:{entry['partition']}:{entry['instance_id']}",
        )
        updated = _transform_instance(instance, witness, seed)
        verification = verify_placement(updated, witness)
        if not verification.feasible:
            raise RuntimeError(
                f"Transformed witness is infeasible for {entry['instance_id']}: "
                f"{verification.to_dict()}"
            )
        audit = audit_graph_readiness(updated, witness)
        if not audit.ready:
            raise RuntimeError(
                f"Transformed instance is not graph-ready for {entry['instance_id']}: "
                f"{audit.to_dict()}"
            )
        path = output_root / entry["path"]
        save_instance(updated, path)
        objective = evaluate_latency(updated, witness).objective
        entry["sha256"] = _sha256(path)
        entry["witness_objective"] = float(objective)
        entry["graph_readiness"] = audit.to_dict()
        entry["profile_transform_version"] = TRANSFORM_VERSION
        entry["profile_seed"] = int(seed)
        transformed.append(entry["instance_id"])
        print(f"transformed {entry['instance_id']}", flush=True)

    manifest["generator_version"] = "1.0+realistic_profile_v1"
    manifest["generation_config"]["profile_transform"] = {
        "version": TRANSFORM_VERSION,
        "description": "Scenario-specific synthetic edge-video profile transform.",
    }
    write_json(output_root / "manifest.json", manifest)
    report = {
        "schema_version": "1.0",
        "profile_transform_version": TRANSFORM_VERSION,
        "dataset_root": str(output_root),
        "instances": len(transformed),
        "config": str(config_path),
        "status": "transformed",
    }
    write_json(output_root / "realistic_profile_transform_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/dataset_phase6e_e_realistic_profile.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/datasets/phase6e-e-realistic-profile",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        prepare(
            _resolve(args.config),
            _resolve(args.output_root),
            force=args.force,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
