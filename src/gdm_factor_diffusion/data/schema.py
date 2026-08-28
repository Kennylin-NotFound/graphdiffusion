"""Canonical, validated schema for one deployment-problem instance."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.1"


def _as_array(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_finite_nonnegative(name: str, array: np.ndarray) -> None:
    _require(np.isfinite(array).all(), f"{name} must contain only finite values.")
    _require((array >= 0).all(), f"{name} must be nonnegative.")


@dataclass(slots=True)
class DeploymentInstance:
    """All static data required by the optimizer and learned solver."""

    instance_id: str
    service_type_id: np.ndarray
    service_features: np.ndarray
    service_demand: np.ndarray
    processing_latency: np.ndarray
    compatibility_mask: np.ndarray
    device_type_id: np.ndarray
    device_features: np.ndarray
    device_capacity: np.ndarray
    connectivity: np.ndarray
    link_rate: np.ndarray
    dependency_index: np.ndarray
    dependency_data_volume: np.ndarray
    application_weight: np.ndarray
    application_type_id: np.ndarray
    membership: np.ndarray
    application_dependency_mask: np.ndarray
    sink_mask: np.ndarray
    topological_order: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.instance_id = str(self.instance_id)
        self.service_type_id = _as_array(self.service_type_id, np.int64)
        self.service_features = _as_array(self.service_features, np.float32)
        self.service_demand = _as_array(self.service_demand, np.float32)
        self.processing_latency = _as_array(self.processing_latency, np.float32)
        self.compatibility_mask = _as_array(self.compatibility_mask, np.bool_)
        self.device_type_id = _as_array(self.device_type_id, np.int64)
        self.device_features = _as_array(self.device_features, np.float32)
        self.device_capacity = _as_array(self.device_capacity, np.float32)
        self.connectivity = _as_array(self.connectivity, np.bool_)
        self.link_rate = _as_array(self.link_rate, np.float32)
        self.dependency_index = _as_array(self.dependency_index, np.int64)
        self.dependency_data_volume = _as_array(
            self.dependency_data_volume, np.float32
        )
        self.application_weight = _as_array(self.application_weight, np.float32)
        self.application_type_id = _as_array(self.application_type_id, np.int64)
        self.membership = _as_array(self.membership, np.bool_)
        self.application_dependency_mask = _as_array(
            self.application_dependency_mask, np.bool_
        )
        self.sink_mask = _as_array(self.sink_mask, np.bool_)
        self.topological_order = _as_array(self.topological_order, np.int64)
        self.metadata = dict(self.metadata)
        self.validate()

    @property
    def num_services(self) -> int:
        return int(self.service_features.shape[0])

    @property
    def num_devices(self) -> int:
        return int(self.device_features.shape[0])

    @property
    def num_applications(self) -> int:
        return int(self.membership.shape[0])

    @property
    def num_dependencies(self) -> int:
        return int(self.dependency_index.shape[1])

    @property
    def num_resources(self) -> int:
        return int(self.service_demand.shape[1])

    def validate(self) -> None:
        _require(bool(self.instance_id.strip()), "instance_id must be nonempty.")

        _require(self.service_features.ndim == 2, "service_features must be [M, F_s].")
        _require(self.device_features.ndim == 2, "device_features must be [D, F_d].")
        m = self.num_services
        d = self.num_devices
        _require(m > 0, "An instance must contain at least one service.")
        _require(d > 0, "An instance must contain at least one device.")
        _require(self.service_type_id.shape == (m,), "service_type_id must be [M].")
        _require(self.device_type_id.shape == (d,), "device_type_id must be [D].")
        _require(
            (self.service_type_id >= 0).all(),
            "service_type_id must contain nonnegative dataset-global IDs.",
        )
        _require(
            (self.device_type_id >= 0).all(),
            "device_type_id must contain nonnegative dataset-global IDs.",
        )

        _require(self.service_demand.ndim == 2, "service_demand must be [M, R].")
        _require(self.service_demand.shape[0] == m, "service_demand has wrong M.")
        r = self.num_resources
        _require(r > 0, "An instance must contain at least one resource type.")
        _require(self.device_capacity.shape == (d, r), "device_capacity must be [D, R].")

        _require(
            self.processing_latency.shape == (m, d),
            "processing_latency must be [M, D].",
        )
        _require(
            self.compatibility_mask.shape == (m, d),
            "compatibility_mask must be [M, D].",
        )
        _require(
            self.compatibility_mask.any(axis=1).all(),
            "Every service must have at least one compatible device.",
        )

        for name, array in (
            ("service_features", self.service_features),
            ("device_features", self.device_features),
        ):
            _require(np.isfinite(array).all(), f"{name} must contain only finite values.")

        for name, array in (
            ("service_demand", self.service_demand),
            ("processing_latency", self.processing_latency),
            ("device_capacity", self.device_capacity),
            ("link_rate", self.link_rate),
            ("dependency_data_volume", self.dependency_data_volume),
            ("application_weight", self.application_weight),
        ):
            _require_finite_nonnegative(name, array)

        _require(
            (self.processing_latency[self.compatibility_mask] > 0).all(),
            "Compatible processing latencies must be positive.",
        )
        _require(
            (self.processing_latency[~self.compatibility_mask] == 0).all(),
            "Incompatible processing latencies must use zero placeholders.",
        )

        individually_fitting = (
            self.service_demand[:, None, :] <= self.device_capacity[None, :, :]
        ).all(axis=2)
        _require(
            (self.compatibility_mask & individually_fitting).any(axis=1).all(),
            "Every service must fit on at least one compatible device.",
        )

        _require(self.connectivity.shape == (d, d), "connectivity must be [D, D].")
        _require(self.link_rate.shape == (d, d), "link_rate must be [D, D].")
        _require(
            not np.diag(self.connectivity).any(),
            "connectivity diagonal must be false; colocation is handled separately.",
        )
        _require(
            (np.diag(self.link_rate) == 0).all(),
            "link_rate diagonal must be zero.",
        )
        _require(
            (self.link_rate[~self.connectivity] == 0).all(),
            "Disconnected device pairs must have zero link rate.",
        )
        _require(
            (self.link_rate[self.connectivity] > 0).all(),
            "Connected device pairs must have positive link rates.",
        )

        _require(
            self.dependency_index.ndim == 2 and self.dependency_index.shape[0] == 2,
            "dependency_index must be [2, E].",
        )
        e = self.num_dependencies
        _require(
            self.dependency_data_volume.shape == (e,),
            "dependency_data_volume must be [E].",
        )
        if e:
            source, target = self.dependency_index
            _require(
                ((0 <= self.dependency_index) & (self.dependency_index < m)).all(),
                "dependency_index contains an out-of-range service index.",
            )
            _require((source != target).all(), "Self dependencies are not allowed.")
            edge_pairs = list(zip(source.tolist(), target.tolist(), strict=True))
            _require(len(set(edge_pairs)) == e, "Duplicate dependencies are not allowed.")
            _require(
                (self.dependency_data_volume > 0).all(),
                "Dependency data volumes must be positive.",
            )
            for service in np.unique(source):
                volumes = self.dependency_data_volume[source == service]
                _require(
                    np.allclose(volumes, volumes[0], rtol=1e-6, atol=1e-8),
                    "All outgoing dependencies of a service must use the same data volume.",
                )

        _require(
            self.topological_order.shape == (m,),
            "topological_order must contain every service exactly once.",
        )
        _require(
            np.array_equal(np.sort(self.topological_order), np.arange(m)),
            "topological_order must be a permutation of service indices.",
        )
        position = np.empty(m, dtype=np.int64)
        position[self.topological_order] = np.arange(m)
        if e:
            _require(
                (position[self.dependency_index[0]] < position[self.dependency_index[1]]).all(),
                "topological_order violates a dependency.",
            )

        _require(self.membership.ndim == 2, "membership must be [A, M].")
        a = self.num_applications
        _require(a > 0, "An instance must contain at least one application.")
        _require(
            self.application_type_id.shape == (a,),
            "application_type_id must be [A].",
        )
        _require(
            (self.application_type_id >= 0).all(),
            "application_type_id must contain nonnegative dataset-global IDs.",
        )
        _require(self.membership.shape == (a, m), "membership must be [A, M].")
        _require(self.membership.any(axis=1).all(), "Every application must contain services.")
        _require(
            self.membership.any(axis=0).all(),
            "Every joint-DAG service must belong to at least one application.",
        )
        _require(self.sink_mask.shape == (a, m), "sink_mask must be [A, M].")
        _require(self.sink_mask.any(axis=1).all(), "Every application must have a sink.")
        _require(
            (~self.sink_mask | self.membership).all(),
            "Every sink must belong to its application.",
        )
        _require(
            self.application_dependency_mask.shape == (a, e),
            "application_dependency_mask must be [A, E].",
        )
        if e:
            _require(
                self.application_dependency_mask.any(axis=0).all(),
                "Every joint-DAG dependency must belong to at least one application.",
            )
            source, target = self.dependency_index
            for application in range(a):
                selected = self.application_dependency_mask[application]
                _require(
                    (
                        self.membership[application, source[selected]]
                        & self.membership[application, target[selected]]
                    ).all(),
                    "Application dependency endpoints must belong to that application.",
                )
                outgoing = np.zeros(m, dtype=np.bool_)
                outgoing[source[selected]] = True
                _require(
                    not (self.sink_mask[application] & outgoing).any(),
                    "An application sink cannot have an outgoing dependency in that application.",
                )

        _require(
            self.application_weight.shape == (a,),
            "application_weight must be [A].",
        )
        _require(
            np.isclose(self.application_weight.sum(), 1.0, rtol=1e-6, atol=1e-6),
            "application_weight must sum to one.",
        )

        _require(
            self.metadata.get("schema_version") == SCHEMA_VERSION,
            f"metadata.schema_version must equal {SCHEMA_VERSION!r}.",
        )
        try:
            json.dumps(self.metadata, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be JSON serializable.") from error

        for key, expected in (
            ("service_feature_names", self.service_features.shape[1]),
            ("device_feature_names", self.device_features.shape[1]),
            ("resource_names", r),
        ):
            if key in self.metadata:
                _require(
                    len(self.metadata[key]) == expected,
                    f"metadata.{key} must contain {expected} names.",
                )

    def equivalent_to(self, other: object) -> bool:
        if not isinstance(other, DeploymentInstance):
            return False
        if self.instance_id != other.instance_id or self.metadata != other.metadata:
            return False
        return all(
            np.array_equal(getattr(self, field.name), getattr(other, field.name))
            for field in fields(self)
            if field.name not in {"instance_id", "metadata"}
        )


_ARRAY_FIELDS = tuple(
    field.name
    for field in fields(DeploymentInstance)
    if field.name not in {"instance_id", "metadata"}
)


def save_instance(instance: DeploymentInstance, path: str | Path) -> Path:
    """Validate and atomically save an instance as a compressed NPZ file."""

    instance.validate()
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("Deployment instances must use the .npz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {name: getattr(instance, name) for name in _ARRAY_FIELDS}
    payload["_instance_id"] = np.asarray(instance.instance_id)
    payload["_metadata_json"] = np.asarray(
        json.dumps(instance.metadata, sort_keys=True, separators=(",", ":"))
    )
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, destination)
    return destination


def load_instance(path: str | Path) -> DeploymentInstance:
    """Load and validate a compressed deployment instance without pickle."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        missing = set(_ARRAY_FIELDS) - set(payload.files)
        _require(not missing, f"Instance file is missing fields: {sorted(missing)}")
        _require("_instance_id" in payload.files, "Instance file is missing _instance_id.")
        _require("_metadata_json" in payload.files, "Instance file is missing _metadata_json.")
        values = {name: payload[name] for name in _ARRAY_FIELDS}
        instance_id = str(payload["_instance_id"].item())
        metadata = json.loads(str(payload["_metadata_json"].item()))
    return DeploymentInstance(instance_id=instance_id, metadata=metadata, **values)
