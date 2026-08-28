"""Constructively feasible, graph-ready deployment-instance generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gdm_factor_diffusion.solver.latency_evaluator import evaluate_latency
from gdm_factor_diffusion.solver.placement_verifier import verify_placement

from .catalogs import (
    APPLICATION_BY_ID,
    APPLICATION_TYPES,
    CATALOG_VERSION,
    DEVICE_BY_ID,
    DEVICE_TYPES,
    SERVICE_BY_ID,
)
from .schema import SCHEMA_VERSION, DeploymentInstance


@dataclass(frozen=True, slots=True)
class InstanceGenerationSpec:
    instance_id: str
    seed: int
    partition: str
    role: str
    regime: str
    size_profile: str
    num_applications: int
    num_devices: int
    share_probability: float
    compatibility_density: float
    topology_density: float
    capacity_slack: float
    minimum_candidates: int = 2
    application_type_ids: tuple[int, ...] | None = None
    application_type_pool: tuple[int, ...] | None = None

    def validate(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id must be nonempty.")
        if self.num_applications < 1:
            raise ValueError("num_applications must be positive.")
        if self.num_devices < 1:
            raise ValueError("num_devices must be positive.")
        for name, value in (
            ("share_probability", self.share_probability),
            ("compatibility_density", self.compatibility_density),
            ("topology_density", self.topology_density),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.capacity_slack < 0:
            raise ValueError("capacity_slack must be nonnegative.")
        if self.minimum_candidates < 1:
            raise ValueError("minimum_candidates must be positive.")
        if self.application_type_ids is not None:
            if len(self.application_type_ids) != self.num_applications:
                raise ValueError(
                    "application_type_ids length must equal num_applications."
                )
            unknown = set(self.application_type_ids) - set(APPLICATION_BY_ID)
            if unknown:
                raise ValueError(f"Unknown application type IDs: {sorted(unknown)}")
        if self.application_type_pool is not None:
            if self.application_type_ids is not None:
                raise ValueError(
                    "Use application_type_ids or application_type_pool, not both."
                )
            if not self.application_type_pool:
                raise ValueError("application_type_pool must be nonempty.")
            unknown = set(self.application_type_pool) - set(APPLICATION_BY_ID)
            if unknown:
                raise ValueError(f"Unknown application type IDs: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class GeneratedInstance:
    instance: DeploymentInstance
    witness_placement: np.ndarray
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Workflow:
    service_type_id: np.ndarray
    dependency_index: np.ndarray
    application_type_id: np.ndarray
    membership: np.ndarray
    application_dependency_mask: np.ndarray
    sink_mask: np.ndarray
    topological_order: np.ndarray
    total_local_services: int


def _select_application_types(
    spec: InstanceGenerationSpec,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    if spec.application_type_ids is not None:
        return spec.application_type_ids
    available = np.asarray(
        spec.application_type_pool
        if spec.application_type_pool is not None
        else [item.type_id for item in APPLICATION_TYPES],
        dtype=np.int64,
    )
    replace = spec.num_applications > len(available)
    return tuple(
        int(value)
        for value in rng.choice(available, size=spec.num_applications, replace=replace)
    )


def _build_workflow(
    spec: InstanceGenerationSpec,
    rng: np.random.Generator,
) -> _Workflow:
    application_type_ids = _select_application_types(spec, rng)
    global_type_ids: list[int] = []
    merge_candidates: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    application_nodes: list[set[int]] = []
    application_edges: list[set[tuple[int, int]]] = []
    application_sinks: list[set[int]] = []
    joint_edges: set[tuple[int, int]] = set()
    total_local_services = 0

    for application_type_id in application_type_ids:
        template = APPLICATION_BY_ID[application_type_id]
        total_local_services += len(template.service_type_ids)
        predecessors: list[list[int]] = [
            [] for _ in range(len(template.service_type_ids))
        ]
        for source, target in template.edges:
            predecessors[target].append(source)

        local_to_global: dict[int, int] = {}
        for local_index, service_type_id in enumerate(template.service_type_ids):
            global_predecessors = tuple(
                sorted(local_to_global[index] for index in predecessors[local_index])
            )
            merge_key = (service_type_id, global_predecessors)
            candidates = merge_candidates.get(merge_key, [])
            if candidates and rng.random() < spec.share_probability:
                global_index = int(rng.choice(candidates))
            else:
                global_index = len(global_type_ids)
                global_type_ids.append(service_type_id)
                merge_candidates.setdefault(merge_key, []).append(global_index)
            local_to_global[local_index] = global_index

        nodes = set(local_to_global.values())
        edges = {
            (local_to_global[source], local_to_global[target])
            for source, target in template.edges
        }
        sinks = {local_to_global[index] for index in template.sink_indices}
        application_nodes.append(nodes)
        application_edges.append(edges)
        application_sinks.append(sinks)
        joint_edges.update(edges)

    sorted_edges = sorted(joint_edges)
    edge_to_index = {edge: index for index, edge in enumerate(sorted_edges)}
    m = len(global_type_ids)
    e = len(sorted_edges)
    a = len(application_type_ids)
    membership = np.zeros((a, m), dtype=np.bool_)
    application_dependency_mask = np.zeros((a, e), dtype=np.bool_)
    sink_mask = np.zeros((a, m), dtype=np.bool_)
    for application in range(a):
        membership[application, list(application_nodes[application])] = True
        sink_mask[application, list(application_sinks[application])] = True
        application_dependency_mask[
            application,
            [edge_to_index[edge] for edge in application_edges[application]],
        ] = True

    dependency_index = (
        np.asarray(sorted_edges, dtype=np.int64).T
        if sorted_edges
        else np.empty((2, 0), dtype=np.int64)
    )
    return _Workflow(
        service_type_id=np.asarray(global_type_ids, dtype=np.int64),
        dependency_index=dependency_index,
        application_type_id=np.asarray(application_type_ids, dtype=np.int64),
        membership=membership,
        application_dependency_mask=application_dependency_mask,
        sink_mask=sink_mask,
        topological_order=np.arange(m, dtype=np.int64),
        total_local_services=total_local_services,
    )


def _sample_service_quantities(
    workflow: _Workflow,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = len(workflow.service_type_id)
    processing_demand = np.zeros(m, dtype=np.float32)
    output_volume = np.zeros(m, dtype=np.float32)
    service_demand = np.zeros((m, 2), dtype=np.float32)
    stage = np.zeros(m, dtype=np.float32)
    for service, type_id in enumerate(workflow.service_type_id):
        template = SERVICE_BY_ID[int(type_id)]
        processing_demand[service] = template.processing_demand * rng.uniform(0.9, 1.1)
        output_volume[service] = template.output_data_volume * rng.uniform(0.9, 1.1)
        service_demand[service] = np.asarray(template.resource_demand) * rng.uniform(
            0.9, 1.1, size=2
        )
        stage[service] = template.stage
    return processing_demand, output_volume, service_demand, stage


def _sample_device_types(
    num_devices: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_devices >= len(DEVICE_TYPES):
        type_ids = [item.type_id for item in DEVICE_TYPES]
        remaining = num_devices - len(type_ids)
        if remaining:
            type_ids.extend(
                rng.choice([0, 1, 2], size=remaining, p=[0.30, 0.45, 0.25]).tolist()
            )
        rng.shuffle(type_ids)
        return np.asarray(type_ids, dtype=np.int64)
    type_ids = rng.choice([1, 2], size=num_devices, replace=True, p=[0.65, 0.35])
    type_ids[0] = 2
    rng.shuffle(type_ids)
    return np.asarray(type_ids, dtype=np.int64)


def _construct_witness(
    service_type_id: np.ndarray,
    service_demand: np.ndarray,
    device_type_id: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    d = len(device_type_id)
    base_capacity = np.asarray(
        [DEVICE_BY_ID[int(type_id)].base_capacity for type_id in device_type_id],
        dtype=np.float64,
    )
    device_level = np.asarray(
        [DEVICE_BY_ID[int(type_id)].level for type_id in device_type_id],
        dtype=np.int64,
    )
    current_load = np.zeros((d, 2), dtype=np.float64)
    witness = np.empty(len(service_type_id), dtype=np.int64)
    service_order = sorted(
        range(len(service_type_id)),
        key=lambda index: float(service_demand[index].sum()),
        reverse=True,
    )
    for service in service_order:
        minimum_level = SERVICE_BY_ID[int(service_type_id[service])].minimum_device_level
        candidates = np.flatnonzero(device_level >= minimum_level)
        if len(candidates) == 0:
            raise RuntimeError("No device type can host a generated service.")
        projected = current_load[candidates] + service_demand[service]
        score = np.max(projected / base_capacity[candidates], axis=1)
        score += rng.uniform(0, 0.05, size=len(candidates))
        selected = int(candidates[int(np.argmin(score))])
        witness[service] = selected
        current_load[selected] += service_demand[service]
    return witness


def _build_capacity(
    witness: np.ndarray,
    service_demand: np.ndarray,
    device_type_id: np.ndarray,
    capacity_slack: float,
    rng: np.random.Generator,
) -> np.ndarray:
    d = len(device_type_id)
    witness_load = np.zeros((d, 2), dtype=np.float64)
    for service, device in enumerate(witness):
        witness_load[device] += service_demand[service]
    base_capacity = np.asarray(
        [DEVICE_BY_ID[int(type_id)].base_capacity for type_id in device_type_id],
        dtype=np.float64,
    )
    floor = base_capacity * rng.uniform(0.30, 0.45, size=(d, 1))
    capacity = np.maximum(witness_load * (1.0 + capacity_slack), floor)
    capacity = np.maximum(capacity, witness_load + 1e-3)
    return capacity.astype(np.float32)


def _build_compatibility(
    spec: InstanceGenerationSpec,
    service_type_id: np.ndarray,
    device_type_id: np.ndarray,
    witness: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    m = len(service_type_id)
    d = len(device_type_id)
    device_level = np.asarray(
        [DEVICE_BY_ID[int(type_id)].level for type_id in device_type_id],
        dtype=np.int64,
    )
    compatibility = np.zeros((m, d), dtype=np.bool_)
    for service, type_id in enumerate(service_type_id):
        minimum_level = SERVICE_BY_ID[int(type_id)].minimum_device_level
        hardware_candidates = np.flatnonzero(device_level >= minimum_level)
        sampled = hardware_candidates[
            rng.random(len(hardware_candidates)) < spec.compatibility_density
        ]
        compatibility[service, sampled] = True
        compatibility[service, witness[service]] = True
        desired = min(spec.minimum_candidates, len(hardware_candidates))
        missing = desired - int(compatibility[service].sum())
        if missing > 0:
            remaining = hardware_candidates[~compatibility[service, hardware_candidates]]
            chosen = rng.choice(remaining, size=missing, replace=False)
            compatibility[service, chosen] = True
    return compatibility


def _build_topology(
    spec: InstanceGenerationSpec,
    dependency_index: np.ndarray,
    witness: np.ndarray,
    device_type_id: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    d = len(device_type_id)
    forced_edges = {
        tuple(sorted((int(witness[source]), int(witness[target]))))
        for source, target in dependency_index.T
        if witness[source] != witness[target]
    }
    all_edges = [(left, right) for left in range(d) for right in range(left + 1, d)]
    target_edges = int(np.ceil(spec.topology_density * len(all_edges)))
    selected_edges = set(forced_edges)
    remaining = [edge for edge in all_edges if edge not in selected_edges]
    rng.shuffle(remaining)
    selected_edges.update(remaining[: max(0, target_edges - len(selected_edges))])

    connectivity = np.zeros((d, d), dtype=np.bool_)
    link_rate = np.zeros((d, d), dtype=np.float32)
    for left, right in selected_edges:
        left_level = DEVICE_BY_ID[int(device_type_id[left])].level
        right_level = DEVICE_BY_ID[int(device_type_id[right])].level
        rate = rng.uniform(30.0, 80.0) * (1.0 + 0.45 * min(left_level, right_level))
        connectivity[left, right] = connectivity[right, left] = True
        link_rate[left, right] = link_rate[right, left] = float(rate)
    return connectivity, link_rate


def _edge_degrees(num_services: int, dependency_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    in_degree = np.zeros(num_services, dtype=np.float32)
    out_degree = np.zeros(num_services, dtype=np.float32)
    if dependency_index.shape[1]:
        np.add.at(out_degree, dependency_index[0], 1)
        np.add.at(in_degree, dependency_index[1], 1)
    return in_degree, out_degree


def generate_instance(spec: InstanceGenerationSpec) -> GeneratedInstance:
    """Generate one deterministic-by-seed instance with a verified witness."""

    spec.validate()
    rng = np.random.default_rng(spec.seed)
    workflow = _build_workflow(spec, rng)
    processing_demand, output_volume, service_demand, stage = _sample_service_quantities(
        workflow, rng
    )
    device_type_id = _sample_device_types(spec.num_devices, rng)
    witness = _construct_witness(
        workflow.service_type_id, service_demand, device_type_id, rng
    )
    device_capacity = _build_capacity(
        witness, service_demand, device_type_id, spec.capacity_slack, rng
    )
    compatibility = _build_compatibility(
        spec, workflow.service_type_id, device_type_id, witness, rng
    )
    connectivity, link_rate = _build_topology(
        spec, workflow.dependency_index, witness, device_type_id, rng
    )

    device_frequency = np.asarray(
        [
            DEVICE_BY_ID[int(type_id)].processing_frequency * rng.uniform(0.9, 1.1)
            for type_id in device_type_id
        ],
        dtype=np.float32,
    )
    processing_latency = np.zeros_like(compatibility, dtype=np.float32)
    all_latency = processing_demand[:, None] / device_frequency[None, :]
    processing_latency[compatibility] = all_latency[compatibility]

    application_count = workflow.membership.sum(axis=0).astype(np.float32)
    in_degree, out_degree = _edge_degrees(
        len(workflow.service_type_id), workflow.dependency_index
    )
    service_features = np.column_stack(
        (
            processing_demand,
            output_volume,
            service_demand,
            stage,
            application_count,
            application_count > 1,
            in_degree,
            out_degree,
        )
    ).astype(np.float32)

    degree = connectivity.sum(axis=1).astype(np.float32)
    nonzero_rate_count = np.maximum(degree, 1)
    mean_link_rate = link_rate.sum(axis=1) / nonzero_rate_count
    device_features = np.column_stack(
        (
            device_frequency,
            device_capacity,
            degree,
            mean_link_rate,
        )
    ).astype(np.float32)

    source = workflow.dependency_index[0]
    dependency_data_volume = output_volume[source].astype(np.float32)
    application_weight = np.full(
        len(workflow.application_type_id),
        1.0 / len(workflow.application_type_id),
        dtype=np.float32,
    )

    total_possible_links = spec.num_devices * (spec.num_devices - 1)
    actual_topology_density = (
        float(connectivity.sum() / total_possible_links)
        if total_possible_links
        else 0.0
    )
    actual_compatibility_density = float(compatibility.mean())
    sharing_ratio = float(
        1.0 - len(workflow.service_type_id) / workflow.total_local_services
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "catalog_version": CATALOG_VERSION,
        "generator_version": "1.0",
        "partition": spec.partition,
        "role": spec.role,
        "regime": spec.regime,
        "size_profile": spec.size_profile,
        "generation_seed": int(spec.seed),
        "input_available_at_all_devices": True,
        "multi_hop_forwarding": False,
        "processing_contention_modeled": False,
        "symmetric_direct_links": True,
        "link_rate_representation": "effective_direct_rate",
        "quantity_units": {
            "processing_demand": "normalized_compute_work",
            "processing_frequency": "normalized_compute_rate",
            "processing_latency": "normalized_time",
            "output_data_volume": "normalized_data",
            "link_rate": "normalized_data_per_time",
            "resource_demand": "normalized_resource_units",
            "device_capacity": "normalized_resource_units",
        },
        "requested_share_probability": spec.share_probability,
        "requested_compatibility_density": spec.compatibility_density,
        "requested_topology_density": spec.topology_density,
        "capacity_slack": spec.capacity_slack,
        "actual_sharing_ratio": sharing_ratio,
        "actual_compatibility_density": actual_compatibility_density,
        "actual_topology_density": actual_topology_density,
        "service_feature_names": [
            "processing_demand",
            "output_data_volume",
            "cpu_demand",
            "memory_demand",
            "workflow_stage",
            "application_count",
            "is_shared",
            "in_degree",
            "out_degree",
        ],
        "device_feature_names": [
            "processing_frequency",
            "cpu_capacity",
            "memory_capacity",
            "direct_link_degree",
            "mean_direct_link_rate",
        ],
        "resource_names": ["cpu_units", "memory_units"],
    }
    instance = DeploymentInstance(
        instance_id=spec.instance_id,
        service_type_id=workflow.service_type_id,
        service_features=service_features,
        service_demand=service_demand,
        processing_latency=processing_latency,
        compatibility_mask=compatibility,
        device_type_id=device_type_id,
        device_features=device_features,
        device_capacity=device_capacity,
        connectivity=connectivity,
        link_rate=link_rate,
        dependency_index=workflow.dependency_index,
        dependency_data_volume=dependency_data_volume,
        application_weight=application_weight,
        application_type_id=workflow.application_type_id,
        membership=workflow.membership,
        application_dependency_mask=workflow.application_dependency_mask,
        sink_mask=workflow.sink_mask,
        topological_order=workflow.topological_order,
        metadata=metadata,
    )
    verification = verify_placement(instance, witness)
    if not verification.feasible:
        raise RuntimeError(
            f"Constructed witness is infeasible: {verification.to_dict()}"
        )
    witness_objective = evaluate_latency(instance, witness).objective
    summary = {
        "instance_id": spec.instance_id,
        "partition": spec.partition,
        "role": spec.role,
        "regime": spec.regime,
        "size_profile": spec.size_profile,
        "seed": int(spec.seed),
        "num_services": instance.num_services,
        "num_devices": instance.num_devices,
        "num_applications": instance.num_applications,
        "num_dependencies": instance.num_dependencies,
        "application_type_ids": instance.application_type_id.tolist(),
        "sharing_ratio": sharing_ratio,
        "compatibility_density": actual_compatibility_density,
        "topology_density": actual_topology_density,
        "witness_placement": witness.tolist(),
        "witness_objective": witness_objective,
        "witness_is_model_input": False,
    }
    return GeneratedInstance(instance=instance, witness_placement=witness, summary=summary)
