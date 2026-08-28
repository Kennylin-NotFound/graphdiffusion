"""Dataset partition generation, manifests, and leakage-resistant loading."""

from __future__ import annotations

import json
import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed

from .catalogs import CATALOG_VERSION, catalog_as_dict
from .contracts import audit_dataset_config_contract
from .generator import InstanceGenerationSpec, generate_instance
from .graph_readiness import audit_graph_readiness
from .schema import SCHEMA_VERSION, DeploymentInstance, load_instance, save_instance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_int_range(rng: np.random.Generator, value: Any, name: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, list) and len(value) == 2:
        low, high = (int(item) for item in value)
        if low > high:
            raise ValueError(f"{name} range must be ordered.")
        return int(rng.integers(low, high + 1))
    raise TypeError(f"{name} must be an integer or [low, high].")


def _sample_float_range(rng: np.random.Generator, value: Any, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and len(value) == 2:
        low, high = (float(item) for item in value)
        if low > high:
            raise ValueError(f"{name} range must be ordered.")
        return float(rng.uniform(low, high))
    raise TypeError(f"{name} must be a number or [low, high].")


def _build_spec(
    dataset_name: str,
    base_seed: int,
    partition_name: str,
    partition: Mapping[str, Any],
    index: int,
    attempt: int = 0,
) -> InstanceGenerationSpec:
    seed = derive_seed(
        base_seed,
        f"{dataset_name}:{partition_name}:{index}:attempt={attempt}",
    )
    rng = np.random.default_rng(seed)
    application_type_ids = partition.get("application_type_ids")
    application_type_pool = partition.get("application_type_pool")
    return InstanceGenerationSpec(
        instance_id=f"{partition_name}-{index:05d}",
        seed=seed,
        partition=partition_name,
        role=str(partition["role"]),
        regime=str(partition["regime"]),
        size_profile=str(partition["size_profile"]),
        num_applications=_sample_int_range(
            rng, partition["num_applications"], "num_applications"
        ),
        num_devices=_sample_int_range(rng, partition["num_devices"], "num_devices"),
        share_probability=_sample_float_range(
            rng, partition["share_probability"], "share_probability"
        ),
        compatibility_density=_sample_float_range(
            rng, partition["compatibility_density"], "compatibility_density"
        ),
        topology_density=_sample_float_range(
            rng, partition["topology_density"], "topology_density"
        ),
        capacity_slack=_sample_float_range(
            rng, partition["capacity_slack"], "capacity_slack"
        ),
        minimum_candidates=int(partition.get("minimum_candidates", 2)),
        application_type_ids=tuple(int(item) for item in application_type_ids)
        if application_type_ids is not None
        else None,
        application_type_pool=tuple(int(item) for item in application_type_pool)
        if application_type_pool is not None
        else None,
    )


def generate_dataset(
    config: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate all configured partitions and return the saved manifest."""

    dataset = config["dataset"]
    audit_dataset_config_contract(config)
    dataset_name = str(dataset["name"])
    base_seed = int(dataset["base_seed"])
    root = Path(output_root or dataset["output"])
    generation_config = copy.deepcopy(dataset)
    generation_config.pop("output", None)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "catalog.json", catalog_as_dict())

    manifest_entries: list[dict[str, Any]] = []
    used_seeds: set[int] = set()
    for partition_name, partition in dataset["partitions"].items():
        count = int(partition["count"])
        if count < 1:
            raise ValueError(f"Partition {partition_name!r} must have positive count.")
        for index in range(count):
            service_count_range = partition.get("service_count_range")
            if service_count_range is not None:
                if not isinstance(service_count_range, list) or len(service_count_range) != 2:
                    raise TypeError("service_count_range must be [low, high].")
                service_low, service_high = (int(value) for value in service_count_range)
                if service_low < 1 or service_low > service_high:
                    raise ValueError("service_count_range must be positive and ordered.")
            else:
                service_low = service_high = None
            max_attempts = int(partition.get("max_generation_attempts", 100))
            if max_attempts < 1:
                raise ValueError("max_generation_attempts must be positive.")

            generated = None
            accepted_attempt = None
            for attempt in range(max_attempts):
                spec = _build_spec(
                    dataset_name,
                    base_seed,
                    partition_name,
                    partition,
                    index,
                    attempt,
                )
                candidate = generate_instance(spec)
                if service_low is not None and not (
                    service_low <= candidate.instance.num_services <= service_high
                ):
                    continue
                generated = candidate
                accepted_attempt = attempt
                break
            if generated is None or accepted_attempt is None:
                raise RuntimeError(
                    f"Could not generate partition {partition_name!r} index {index} "
                    f"within service_count_range after {max_attempts} attempts."
                )
            if spec.seed in used_seeds:
                raise RuntimeError("Derived generation seed collision.")
            used_seeds.add(spec.seed)

            audit = audit_graph_readiness(
                generated.instance, generated.witness_placement
            )
            if not audit.ready:
                raise RuntimeError(
                    f"Generated instance is not graph-ready: {audit.to_dict()}"
                )
            relative_path = Path(partition_name) / f"{spec.instance_id}.npz"
            saved_path = save_instance(generated.instance, root / relative_path)
            entry = dict(generated.summary)
            entry["generation_attempt"] = accepted_attempt
            entry["path"] = relative_path.as_posix()
            entry["sha256"] = _sha256(saved_path)
            entry["graph_readiness"] = audit.to_dict()
            manifest_entries.append(entry)

    manifest = {
        "dataset_name": dataset_name,
        "base_seed": base_seed,
        "schema_version": SCHEMA_VERSION,
        "catalog_version": CATALOG_VERSION,
        "generator_version": "1.0",
        "instance_count": len(manifest_entries),
        "catalog_path": "catalog.json",
        "generation_config": generation_config,
        "partitions": {
            name: {
                "role": str(partition["role"]),
                "regime": str(partition["regime"]),
                "size_profile": str(partition["size_profile"]),
                "count": int(partition["count"]),
                "service_count_range": partition.get("service_count_range"),
            }
            for name, partition in dataset["partitions"].items()
        },
        "instances": manifest_entries,
    }
    audit_manifest(manifest)
    write_json(root / "manifest.json", manifest)
    return manifest


def audit_manifest(manifest: Mapping[str, Any]) -> None:
    entries = manifest["instances"]
    audit_dataset_config_contract(
        {"dataset": manifest["generation_config"]},
        observed_instance_count=len(entries),
    )
    instance_ids = [entry["instance_id"] for entry in entries]
    seeds = [entry["seed"] for entry in entries]
    paths = [entry["path"] for entry in entries]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("Dataset manifest contains duplicate instance IDs.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Dataset manifest contains duplicate generation seeds.")
    if len(paths) != len(set(paths)):
        raise ValueError("Dataset manifest contains duplicate paths.")
    for entry in entries:
        partition = manifest["partitions"].get(entry["partition"])
        if partition is None:
            raise ValueError(f"Unknown partition in manifest: {entry['partition']}")
        if entry["role"] != partition["role"]:
            raise ValueError("Instance role disagrees with its partition role.")
        if entry["witness_is_model_input"]:
            raise ValueError("Generation witness must never be marked as model input.")
        service_count_range = partition.get("service_count_range")
        if service_count_range is not None and not (
            int(service_count_range[0])
            <= int(entry["num_services"])
            <= int(service_count_range[1])
        ):
            raise ValueError("Instance service count is outside its partition contract.")
        generation_config = manifest.get("generation_config", {})
        configured_partition = generation_config.get("partitions", {}).get(
            entry["partition"], {}
        )
        type_pool = configured_partition.get("application_type_pool")
        if type_pool is not None and not set(entry["application_type_ids"]).issubset(
            set(type_pool)
        ):
            raise ValueError("Instance uses an application type outside its partition pool.")

    generation_partitions = manifest.get("generation_config", {}).get("partitions", {})
    train_type_pool: set[int] = set()
    for partition in generation_partitions.values():
        if partition.get("role") == "train":
            train_type_pool.update(partition.get("application_type_pool", []))
            train_type_pool.update(partition.get("application_type_ids", []))
    for partition in generation_partitions.values():
        if partition.get("regime") == "unseen_workflow":
            held_out = set(partition.get("application_type_pool", []))
            held_out.update(partition.get("application_type_ids", []))
            if not held_out or not held_out.isdisjoint(train_type_pool):
                raise ValueError(
                    "unseen_workflow application types must be nonempty and "
                    "disjoint from the training type pool."
                )


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    audit_manifest(manifest)
    return manifest


def load_partition(
    dataset_root: str | Path,
    partition: str,
    *,
    verify_checksum: bool = True,
) -> list[DeploymentInstance]:
    """Load instance NPZ files only; generation witnesses remain outside model input."""

    root = Path(dataset_root)
    manifest = load_manifest(root / "manifest.json")
    instances: list[DeploymentInstance] = []
    for entry in manifest["instances"]:
        if entry["partition"] != partition:
            continue
        instances.append(
            load_manifest_instance(root, entry, verify_checksum=verify_checksum)
        )
    return instances


def load_manifest_instance(
    dataset_root: str | Path,
    entry: Mapping[str, Any],
    *,
    verify_checksum: bool = True,
) -> DeploymentInstance:
    """Load one manifest entry while checking its immutable content digest."""

    path = Path(dataset_root) / entry["path"]
    if verify_checksum and _sha256(path) != entry["sha256"]:
        raise ValueError(f"Checksum mismatch for dataset instance: {path}")
    return load_instance(path)
