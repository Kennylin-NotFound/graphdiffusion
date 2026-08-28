"""Stable dataset-global service, application, and device type catalogs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CATALOG_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ServiceType:
    type_id: int
    name: str
    stage: float
    processing_demand: float
    output_data_volume: float
    resource_demand: tuple[float, float]
    minimum_device_level: int


@dataclass(frozen=True, slots=True)
class ApplicationType:
    type_id: int
    name: str
    service_type_ids: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    sink_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DeviceType:
    type_id: int
    name: str
    level: int
    processing_frequency: float
    base_capacity: tuple[float, float]


SERVICE_TYPES = (
    ServiceType(0, "decode", 0.0, 8.0, 12.0, (1.0, 2.0), 0),
    ServiceType(1, "resize", 1.0, 10.0, 8.0, (1.0, 2.0), 0),
    ServiceType(2, "detector", 2.0, 35.0, 4.0, (3.0, 6.0), 1),
    ServiceType(3, "tracker", 3.0, 18.0, 2.0, (2.0, 3.0), 1),
    ServiceType(4, "classifier", 3.0, 20.0, 1.5, (2.0, 4.0), 1),
    ServiceType(5, "segmenter", 2.0, 45.0, 5.0, (4.0, 8.0), 2),
    ServiceType(6, "pose", 3.0, 32.0, 3.0, (3.0, 6.0), 1),
    ServiceType(7, "anonymizer", 1.5, 15.0, 8.0, (2.0, 3.0), 0),
    ServiceType(8, "aggregator", 4.0, 12.0, 1.0, (1.0, 2.0), 0),
    ServiceType(9, "alert", 5.0, 8.0, 0.5, (1.0, 1.0), 0),
)

APPLICATION_TYPES = (
    ApplicationType(
        0,
        "tracking",
        (0, 1, 2, 3, 9),
        ((0, 1), (1, 2), (2, 3), (3, 4)),
        (4,),
    ),
    ApplicationType(
        1,
        "classification",
        (0, 1, 2, 4, 9),
        ((0, 1), (1, 2), (2, 3), (3, 4)),
        (4,),
    ),
    ApplicationType(
        2,
        "segmentation",
        (0, 1, 5, 8),
        ((0, 1), (1, 2), (2, 3)),
        (3,),
    ),
    ApplicationType(
        3,
        "pose",
        (0, 1, 6, 8),
        ((0, 1), (1, 2), (2, 3)),
        (3,),
    ),
    ApplicationType(
        4,
        "privacy_tracking",
        (0, 7, 2, 3, 9),
        ((0, 1), (1, 2), (2, 3), (3, 4)),
        (4,),
    ),
    ApplicationType(
        5,
        "branched_detection",
        (0, 1, 2, 3, 4, 8),
        ((0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)),
        (5,),
    ),
)

DEVICE_TYPES = (
    DeviceType(0, "camera", 0, 8.0, (4.0, 8.0)),
    DeviceType(1, "edge_node", 1, 20.0, (14.0, 28.0)),
    DeviceType(2, "edge_server", 2, 40.0, (40.0, 96.0)),
)

SERVICE_BY_ID = {item.type_id: item for item in SERVICE_TYPES}
APPLICATION_BY_ID = {item.type_id: item for item in APPLICATION_TYPES}
DEVICE_BY_ID = {item.type_id: item for item in DEVICE_TYPES}


def catalog_as_dict() -> dict[str, Any]:
    return {
        "catalog_version": CATALOG_VERSION,
        "service_types": [asdict(item) for item in SERVICE_TYPES],
        "application_types": [asdict(item) for item in APPLICATION_TYPES],
        "device_types": [asdict(item) for item in DEVICE_TYPES],
    }
