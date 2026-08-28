from pathlib import Path

import numpy as np
import pytest

from gdm_factor_diffusion.data import (
    InstanceGenerationSpec,
    create_toy_instance,
    generate_dataset,
    generate_instance,
)
from gdm_factor_diffusion.solver import (
    MilpConfig,
    SolutionPoolConfig,
    build_solution_pool,
    enumerate_feasible_placements,
    load_solution_pool,
    save_solution_pool,
    solve_milp,
)
from gdm_factor_diffusion.solver.dataset_labeling import (
    audit_solution_pool,
    audit_solution_pool_manifest,
    label_dataset,
)


def _skip_missing_license(error: Exception) -> None:
    message = str(error).lower()
    if "license" in message or "username" in message:
        pytest.skip(f"Gurobi license is unavailable in this execution context: {error}")
    raise error


def _solve(instance):
    try:
        return solve_milp(instance, MilpConfig(output_flag=False))
    except Exception as error:
        _skip_missing_license(error)


def _pool(instance, *, size: int = 8):
    try:
        return build_solution_pool(
            instance,
            SolutionPoolConfig(
                target_size=size,
                beta=5.0,
                total_time_limit_seconds=60.0,
                output_flag=False,
            ),
        )
    except Exception as error:
        _skip_missing_license(error)


def _small_generated(seed: int):
    return generate_instance(
        InstanceGenerationSpec(
            instance_id=f"milp-random-{seed}",
            seed=seed,
            partition="train",
            role="train",
            regime="in_distribution",
            size_profile="tiny_test",
            num_applications=1,
            num_devices=4,
            share_probability=0.0,
            compatibility_density=0.7,
            topology_density=0.7,
            capacity_slack=0.6,
            minimum_candidates=2,
            application_type_ids=(2,),
        )
    ).instance


def test_milp_optimum_matches_toy_exhaustive_result() -> None:
    instance = create_toy_instance()
    exhaustive = enumerate_feasible_placements(instance)
    result = _solve(instance)

    assert result.optimal
    assert result.status_name == "OPTIMAL"
    assert result.exact_objective == pytest.approx(exhaustive.best_objective)
    assert np.array_equal(result.placement, exhaustive.best_placement)
    assert result.objective_error == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_milp_matches_exhaustive_on_generated_tiny_instances(seed: int) -> None:
    instance = _small_generated(seed)
    exhaustive = enumerate_feasible_placements(instance, max_states=10_000)
    result = _solve(instance)

    assert result.optimal
    assert result.exact_objective == pytest.approx(exhaustive.best_objective)
    assert np.array_equal(result.placement, exhaustive.best_placement)


def test_solution_pool_matches_ordered_toy_exhaustive_prefix(tmp_path: Path) -> None:
    instance = create_toy_instance()
    exhaustive = enumerate_feasible_placements(instance)
    pool = _pool(instance, size=8)

    assert pool.size == 8
    assert np.array_equal(pool.placements, exhaustive.placements[:8])
    assert pool.latencies == pytest.approx(exhaustive.objectives[:8])
    assert np.unique(pool.placements, axis=0).shape[0] == pool.size
    assert pool.sampling_probability.sum() == pytest.approx(1.0)
    assert np.all(pool.sampling_probability[:-1] >= pool.sampling_probability[1:])
    audit_solution_pool(instance, pool)

    path = save_solution_pool(pool, tmp_path / "toy_pool.npz")
    loaded = load_solution_pool(path)
    assert loaded.instance_id == pool.instance_id
    assert np.array_equal(loaded.placements, pool.placements)
    assert loaded.latencies == pytest.approx(pool.latencies)
    audit_solution_pool(instance, loaded)
    loaded.sampling_probability = np.full(loaded.size, 1.0 / loaded.size)
    with pytest.raises(ValueError, match="sampling probabilities"):
        audit_solution_pool(instance, loaded)


def test_dataset_labeling_is_separate_and_auditable(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    config = {
        "dataset": {
            "name": "phase1c-unit",
            "base_seed": 91,
            "output": str(root),
            "partitions": {
                "train": {
                    "role": "train",
                    "regime": "in_distribution",
                    "size_profile": "tiny_test",
                    "count": 1,
                    "num_applications": 1,
                    "num_devices": 3,
                    "share_probability": 0.0,
                    "compatibility_density": 0.3,
                    "topology_density": 0.5,
                    "capacity_slack": 0.4,
                    "minimum_candidates": 1,
                    "application_type_ids": [2],
                }
            },
        }
    }
    generate_dataset(config, output_root=root)
    original_manifest = (root / "manifest.json").read_bytes()
    try:
        manifest = label_dataset(
            root,
            SolutionPoolConfig(
                target_size=3,
                beta=4.0,
                total_time_limit_seconds=60.0,
            ),
        )
    except Exception as error:
        _skip_missing_license(error)

    assert manifest["labeled_instance_count"] == 1
    pool_entry = manifest["pools"][0]
    assert 1 <= pool_entry["pool_size"] <= 3
    if pool_entry["pool_size"] < 3:
        assert pool_entry["termination_reason"] == "infeasible"
    assert (root / "manifest.json").read_bytes() == original_manifest
    audited = audit_solution_pool_manifest(root)
    assert audited == manifest


def test_dataset_labeling_resumes_and_merges_partitions(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    config = {
        "dataset": {
            "name": "phase5a-resume-unit",
            "base_seed": 92,
            "output": str(root),
            "partitions": {
                partition: {
                    "role": partition,
                    "regime": "in_distribution",
                    "size_profile": "tiny_test",
                    "count": 1,
                    "num_applications": 1,
                    "num_devices": 3,
                    "share_probability": 0.0,
                    "compatibility_density": 0.3,
                    "topology_density": 0.5,
                    "capacity_slack": 0.4,
                    "minimum_candidates": 1,
                    "application_type_ids": [2],
                }
                for partition in ("train", "validation")
            },
        }
    }
    generate_dataset(config, output_root=root)
    pool_config = SolutionPoolConfig(
        target_size=2,
        beta=4.0,
        total_time_limit_seconds=60.0,
    )
    try:
        first = label_dataset(root, pool_config, partitions=("train",))
        first_pool_bytes = (root / first["pools"][0]["pool_path"]).read_bytes()
        merged = label_dataset(root, pool_config, partitions=("validation",))
        resumed = label_dataset(root, pool_config, partitions=("train",))
    except Exception as error:
        _skip_missing_license(error)

    assert first["labeled_instance_count"] == 1
    assert merged["labeled_instance_count"] == 2
    assert resumed["labeled_instance_count"] == 2
    assert (root / resumed["pools"][0]["pool_path"]).read_bytes() == first_pool_bytes
    assert audit_solution_pool_manifest(root) == resumed


def test_dataset_labeling_rejects_incompatible_resume_config(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    config = {
        "dataset": {
            "name": "phase5a-config-unit",
            "base_seed": 93,
            "output": str(root),
            "partitions": {
                "train": {
                    "role": "train",
                    "regime": "in_distribution",
                    "size_profile": "tiny_test",
                    "count": 1,
                    "num_applications": 1,
                    "num_devices": 3,
                    "share_probability": 0.0,
                    "compatibility_density": 0.3,
                    "topology_density": 0.5,
                    "capacity_slack": 0.4,
                    "minimum_candidates": 1,
                    "application_type_ids": [2],
                }
            },
        }
    }
    generate_dataset(config, output_root=root)
    try:
        label_dataset(root, SolutionPoolConfig(target_size=2))
    except Exception as error:
        _skip_missing_license(error)

    with pytest.raises(ValueError, match="different labeling configuration"):
        label_dataset(root, SolutionPoolConfig(target_size=3))
