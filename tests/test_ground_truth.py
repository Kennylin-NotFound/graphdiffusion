from dataclasses import replace

import numpy as np
import pytest

from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.solver import (
    InfeasiblePlacementError,
    build_dependency_pair_costs,
    enumerate_feasible_placements,
    evaluate_latency,
    verify_placement,
)


def test_pair_costs_are_finite_masked_and_match_direct_rate() -> None:
    instance = create_toy_instance()
    costs = build_dependency_pair_costs(instance)

    assert costs.admissible.shape == (4, 3, 3)
    assert costs.transmission_latency.shape == (4, 3, 3)
    assert np.isfinite(costs.transmission_latency).all()
    assert costs.admissible[0, 0, 1]
    assert not costs.admissible[0, 0, 2]
    assert costs.transmission_latency[0, 0, 0] == 0
    assert costs.transmission_latency[0, 0, 1] == pytest.approx(8 / 100)


def test_toy_latency_matches_manual_joint_dag_calculation() -> None:
    instance = create_toy_instance()
    placement = np.asarray([0, 1, 1, 0, 1], dtype=np.int64)

    result = evaluate_latency(instance, placement)

    assert result.processing_latency == pytest.approx([1, 1, 0.75, 2.5, 1.5])
    assert result.transmission_latency == pytest.approx([0.08, 0.08, 0.04, 0])
    assert result.completion_time == pytest.approx([1, 2.08, 1.83, 4.62, 3.33])
    assert result.application_latency == pytest.approx([4.62, 3.33])
    assert result.objective == pytest.approx(3.975)
    assert result.critical_sink.tolist() == [3, 4]


def test_binary_and_categorical_placements_verify_identically() -> None:
    instance = create_toy_instance()
    categorical = np.asarray([0, 1, 1, 0, 1], dtype=np.int64)
    binary = np.zeros((instance.num_services, instance.num_devices), dtype=np.int64)
    binary[np.arange(instance.num_services), categorical] = 1

    categorical_report = verify_placement(instance, categorical)
    binary_report = verify_placement(instance, binary)

    assert categorical_report.feasible
    assert binary_report.feasible
    assert np.array_equal(categorical_report.placement, binary_report.placement)
    assert np.array_equal(categorical_report.capacity_load, binary_report.capacity_load)


def test_verifier_separates_assignment_compatibility_capacity_and_link_failures() -> None:
    instance = create_toy_instance()

    bad_assignment = np.zeros((instance.num_services, instance.num_devices))
    assignment_report = verify_placement(instance, bad_assignment)
    assert not assignment_report.assignment_valid
    assert len(assignment_report.assignment_violations) == instance.num_services

    incompatible_report = verify_placement(instance, np.asarray([2, 1, 1, 0, 1]))
    assert not incompatible_report.compatibility_valid
    assert incompatible_report.incompatible_services == (0,)

    reduced_capacity = instance.device_capacity.copy()
    reduced_capacity[1] = [2, 3]
    capacity_instance = replace(instance, device_capacity=reduced_capacity)
    capacity_report = verify_placement(
        capacity_instance, np.asarray([1, 1, 1, 0, 1])
    )
    assert not capacity_report.capacity_valid
    assert capacity_report.total_capacity_excess == pytest.approx(4.0)

    link_report = verify_placement(instance, np.asarray([0, 0, 2, 0, 2]))
    assert not link_report.direct_link_valid
    assert link_report.disconnected_dependencies == (1,)


def test_evaluator_rejects_infeasible_placement() -> None:
    instance = create_toy_instance()
    with pytest.raises(InfeasiblePlacementError) as error:
        evaluate_latency(instance, np.asarray([0, 0, 2, 0, 2]))
    assert error.value.verification.disconnected_dependencies == (1,)


def test_exhaustive_solutions_are_sorted_unique_and_verified() -> None:
    instance = create_toy_instance()
    result = enumerate_feasible_placements(instance)

    assert result.num_candidates == 32
    assert result.num_feasible > 0
    assert np.all(result.objectives[:-1] <= result.objectives[1:])
    assert np.unique(result.placements, axis=0).shape[0] == result.num_feasible
    for placement, objective in zip(result.placements, result.objectives, strict=True):
        assert verify_placement(instance, placement).feasible
        assert evaluate_latency(instance, placement).objective == pytest.approx(objective)
