from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gdm_factor_diffusion.data import create_toy_instance, load_instance, save_instance


def test_toy_instance_round_trip(tmp_path: Path) -> None:
    instance = create_toy_instance()
    path = save_instance(instance, tmp_path / "toy.npz")
    loaded = load_instance(path)

    assert instance.equivalent_to(loaded)
    assert loaded.num_services == 5
    assert loaded.num_devices == 3
    assert loaded.num_applications == 2
    assert loaded.num_dependencies == 4
    assert loaded.num_resources == 2


def test_schema_rejects_incompatible_nonzero_latency() -> None:
    instance = create_toy_instance()
    invalid_latency = instance.processing_latency.copy()
    invalid_latency[~instance.compatibility_mask] = 1

    with pytest.raises(ValueError, match="zero placeholders"):
        replace(instance, processing_latency=invalid_latency)


def test_schema_rejects_dependency_without_application_provenance() -> None:
    instance = create_toy_instance()
    invalid_mask = instance.application_dependency_mask.copy()
    invalid_mask[:, 0] = False

    with pytest.raises(ValueError, match="at least one application"):
        replace(instance, application_dependency_mask=invalid_mask)


def test_schema_rejects_inconsistent_source_data_volume() -> None:
    instance = create_toy_instance()
    invalid_volume = instance.dependency_data_volume.copy()
    invalid_volume[1] += 1

    with pytest.raises(ValueError, match="same data volume"):
        replace(instance, dependency_data_volume=invalid_volume)


def test_schema_rejects_invalid_topological_order() -> None:
    instance = create_toy_instance()
    invalid_order = np.asarray([1, 0, 2, 3, 4], dtype=np.int64)

    with pytest.raises(ValueError, match="violates a dependency"):
        replace(instance, topological_order=invalid_order)
