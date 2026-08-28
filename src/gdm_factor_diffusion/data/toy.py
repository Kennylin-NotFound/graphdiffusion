"""Hand-checkable toy instance used by Phase 0 and Phase 1 tests."""

from __future__ import annotations

import numpy as np

from .schema import SCHEMA_VERSION, DeploymentInstance


def create_toy_instance(instance_id: str = "phase0-toy") -> DeploymentInstance:
    """Create two applications with one shared upstream microservice."""

    processing_demand = np.asarray([10, 20, 15, 25, 30], dtype=np.float32)
    output_data_volume = np.asarray([8, 4, 6, 3, 3], dtype=np.float32)
    application_count = np.asarray([2, 1, 1, 1, 1], dtype=np.float32)
    service_features = np.column_stack(
        (processing_demand, output_data_volume, application_count)
    )

    service_demand = np.asarray(
        [
            [1, 1],
            [1, 1],
            [1, 2],
            [1, 1],
            [1, 1],
        ],
        dtype=np.float32,
    )
    device_capacity = np.asarray([[4, 5], [4, 6], [3, 4]], dtype=np.float32)
    device_frequency = np.asarray([10, 20, 30], dtype=np.float32)

    connectivity = np.asarray(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ]
    )
    link_rate = np.asarray(
        [
            [0, 100, 0],
            [100, 0, 50],
            [0, 50, 0],
        ],
        dtype=np.float32,
    )
    degree = connectivity.sum(axis=1, dtype=np.int64).astype(np.float32)
    device_features = np.column_stack(
        (device_frequency, degree, device_capacity.sum(axis=1))
    )

    compatibility_mask = np.asarray(
        [
            [True, True, False],
            [True, True, False],
            [False, True, True],
            [True, False, True],
            [False, True, True],
        ]
    )
    processing_latency = np.zeros_like(compatibility_mask, dtype=np.float32)
    candidate_latency = processing_demand[:, None] / device_frequency[None, :]
    processing_latency[compatibility_mask] = candidate_latency[compatibility_mask]

    dependency_index = np.asarray(
        [
            [0, 0, 1, 2],
            [1, 2, 3, 4],
        ],
        dtype=np.int64,
    )
    dependency_data_volume = np.asarray([8, 8, 4, 6], dtype=np.float32)

    membership = np.asarray(
        [
            [True, True, False, True, False],
            [True, False, True, False, True],
        ]
    )
    application_dependency_mask = np.asarray(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    sink_mask = np.asarray(
        [
            [False, False, False, True, False],
            [False, False, False, False, True],
        ]
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "description": "Hand-checkable shared-upstream two-application DAG.",
        "input_available_at_all_devices": True,
        "multi_hop_forwarding": False,
        "processing_contention_modeled": False,
        "service_feature_names": [
            "processing_demand",
            "output_data_volume",
            "application_count",
        ],
        "device_feature_names": [
            "processing_frequency",
            "direct_link_degree",
            "total_capacity",
        ],
        "resource_names": ["cpu_units", "memory_units"],
    }

    return DeploymentInstance(
        instance_id=instance_id,
        service_type_id=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        service_features=service_features,
        service_demand=service_demand,
        processing_latency=processing_latency,
        compatibility_mask=compatibility_mask,
        device_type_id=np.asarray([0, 1, 2], dtype=np.int64),
        device_features=device_features,
        device_capacity=device_capacity,
        connectivity=connectivity,
        link_rate=link_rate,
        dependency_index=dependency_index,
        dependency_data_volume=dependency_data_volume,
        application_weight=np.asarray([0.5, 0.5], dtype=np.float32),
        application_type_id=np.asarray([0, 1], dtype=np.int64),
        membership=membership,
        application_dependency_mask=application_dependency_mask,
        sink_mask=sink_mask,
        topological_order=np.arange(5, dtype=np.int64),
        metadata=metadata,
    )
