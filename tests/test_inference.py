import numpy as np
import torch

from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.diffusion import CategoricalSchedule
from gdm_factor_diffusion.graph import build_factor_graph_batch, infer_feature_schema
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    TrajectoryDiagnosticConfig,
    build_trajectory_candidate_set,
    construct_latency_aware_heuristic_candidates,
    construct_feasible_placement,
    construct_greedy_local_placement,
    diagnose_reverse_trajectory,
    run_local_search,
    repair_placement,
    sample_direct_proposals,
    sample_random_proposals,
    sample_reverse_proposals,
    sample_reverse_trajectory_proposals,
    solve_fallback_only,
    solve_from_proposals,
    solve_greedy_local,
    solve_latency_aware_heuristic,
    solve_local_search,
    solve_milp_time_limit,
    violation_score,
)
from gdm_factor_diffusion.models import (
    DenoiserConfig,
    DirectPredictorConfig,
    TypedFactorDenoiser,
    TypedFactorDirectPredictor,
)
from gdm_factor_diffusion.solver import verify_placement


def test_bounded_repair_strictly_reduces_violation_score() -> None:
    instance = create_toy_instance()
    placement = np.asarray([0, 0, 2, 0, 2], dtype=np.int64)

    result = repair_placement(instance, placement)

    assert result.success
    assert result.moves
    assert result.final_verification.feasible
    assert violation_score(instance, result.placement) < violation_score(
        instance, placement
    )
    for move in result.moves:
        assert move.score_after < move.score_before


def test_constructive_fallback_is_deterministic_and_verified() -> None:
    instance = create_toy_instance()

    first = construct_feasible_placement(instance)
    second = construct_feasible_placement(instance)

    assert first.success
    assert second.success
    assert np.array_equal(first.placement, second.placement)
    assert first.search_nodes == second.search_nodes
    assert verify_placement(instance, first.placement).feasible


def test_random_proposals_preserve_assignment_and_compatibility() -> None:
    instance = create_toy_instance()
    proposals = sample_random_proposals(
        instance,
        num_samples=20,
        generator=torch.Generator().manual_seed(11),
    )

    assert proposals.shape == (20, instance.num_services)
    for proposal in proposals:
        report = verify_placement(instance, proposal)
        assert report.assignment_valid
        assert report.compatibility_valid


def test_reverse_proposals_preserve_assignment_and_compatibility() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    model = TypedFactorDenoiser.from_batch(
        batch,
        DenoiserConfig(num_diffusion_steps=3, hidden_dim=16, num_layers=1),
    )
    config = InferenceConfig(num_samples=3, sample_batch_size=2)

    proposals, probabilities, elapsed = sample_reverse_proposals(
        model,
        instance,
        CategoricalSchedule.linear(3, beta_end=0.3),
        infer_feature_schema([instance]),
        config=config,
        generator=torch.Generator().manual_seed(12),
    )

    assert proposals.shape == (3, instance.num_services)
    assert probabilities.shape == (
        3,
        instance.num_services,
        instance.num_devices,
    )
    assert elapsed >= 0
    for proposal in proposals:
        report = verify_placement(instance, proposal)
        assert report.assignment_valid
        assert report.compatibility_valid


def test_reverse_trajectory_diagnostics_are_finite_and_validation_ready() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    model = TypedFactorDenoiser.from_batch(
        batch,
        DenoiserConfig(num_diffusion_steps=3, hidden_dim=16, num_layers=1),
    )
    result = diagnose_reverse_trajectory(
        model,
        instance,
        CategoricalSchedule.linear(3, beta_end=0.3),
        infer_feature_schema([instance]),
        reference_objective=1.0,
        config=TrajectoryDiagnosticConfig(
            num_samples=3,
            reverse_steps=3,
            anchor_count=2,
        ),
        generator=torch.Generator().manual_seed(12),
    )

    assert result["instance_id"] == instance.instance_id
    assert result["anchor_count"] == 2
    assert result["final"]["candidate_count"] == 3
    assert result["reservoir"]["candidate_count"] == 12
    assert result["diagnostic_seconds"] >= 0
    for snapshot in result["snapshots"]:
        assert np.isfinite(list(snapshot["state_use"].values())).all()


def test_trajectory_proposals_reuse_reverse_forwards_and_deduplicate() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    base = TypedFactorDenoiser.from_batch(
        batch,
        DenoiserConfig(num_diffusion_steps=3, hidden_dim=16, num_layers=1),
    )

    class CountingModel(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model
            self.calls = 0

        def forward(self, *args):
            self.calls += 1
            return self.model(*args)

    model = CountingModel(base)
    trajectory = sample_reverse_trajectory_proposals(
        model,
        instance,
        CategoricalSchedule.linear(3, beta_end=0.3),
        infer_feature_schema([instance]),
        anchor_indices=(0, 2),
        config=InferenceConfig(num_samples=3, sample_batch_size=3, reverse_steps=3),
        generator=torch.Generator().manual_seed(21),
    )

    assert model.calls == 4  # Three reverse transitions plus final probabilities.
    assert trajectory.final_proposals.shape == (3, instance.num_services)
    assert set(trajectory.clean_proposals) == {0, 2}
    candidates = build_trajectory_candidate_set(
        trajectory,
        anchor_indices=(0, 2),
    )
    assert candidates.candidates_before_deduplication == 9
    assert candidates.proposals.shape[0] <= 9
    assert candidates.probabilities.shape[0] == candidates.proposals.shape[0]
    assert len(candidates.sources) == candidates.proposals.shape[0]


def test_solver_limits_repair_to_best_ranked_infeasible_candidates() -> None:
    instance = create_toy_instance()
    proposals = np.asarray(
        [
            [0, 0, 2, 0, 2],
            [0, 0, 2, 0, 1],
            [0, 0, 1, 0, 2],
        ],
        dtype=np.int64,
    )
    result = solve_from_proposals(
        instance,
        proposals,
        config=InferenceConfig(
            num_samples=3,
            sample_batch_size=3,
            repair_candidate_limit=1,
            enable_fallback=False,
        ),
    )

    assert result.metrics["repair_attempts"] <= 1
    assert result.metrics["inference_config"]["repair_candidate_limit"] == 1


def test_direct_proposals_preserve_assignment_and_compatibility() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    model = TypedFactorDirectPredictor.from_batch(
        batch,
        DirectPredictorConfig(hidden_dim=16, num_layers=1),
    )
    config = InferenceConfig(num_samples=3, sample_batch_size=2)

    proposals, probabilities, elapsed = sample_direct_proposals(
        model,
        instance,
        infer_feature_schema([instance]),
        config=config,
        generator=torch.Generator().manual_seed(13),
    )

    assert proposals.shape == (3, instance.num_services)
    assert probabilities.shape == (
        3,
        instance.num_services,
        instance.num_devices,
    )
    assert elapsed >= 0
    for proposal in proposals:
        report = verify_placement(instance, proposal)
        assert report.assignment_valid
        assert report.compatibility_valid


def test_solver_returns_only_a_finally_verified_placement() -> None:
    instance = create_toy_instance()
    proposals = np.asarray(
        [
            [0, 0, 2, 0, 2],
            [1, 1, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    result = solve_from_proposals(
        instance,
        proposals,
        config=InferenceConfig(
            num_samples=2,
            sample_batch_size=2,
            repair_max_moves=5,
        ),
        proposal_method="unit",
    )

    assert result.success
    assert verify_placement(instance, result.placement).feasible
    assert result.metrics["num_raw_proposals"] == 2
    assert result.metrics["final_success"]
    assert result.objective is not None


def test_solver_records_pre_fallback_proposal_quality_and_diversity() -> None:
    instance = create_toy_instance()
    proposals = np.asarray(
        [
            [0, 1, 1, 0, 1],
            [0, 1, 1, 0, 1],
            [1, 1, 1, 0, 1],
        ],
        dtype=np.int64,
    )

    result = solve_from_proposals(
        instance,
        proposals,
        config=InferenceConfig(
            num_samples=3,
            sample_batch_size=3,
            enable_repair=False,
            enable_fallback=False,
        ),
        proposal_method="diagnostic-unit",
    )

    assert result.metrics["raw_unique_count"] == 2
    assert result.metrics["raw_unique_rate"] == 2 / 3
    assert result.metrics["raw_pairwise_hamming"] == 2 / 15
    assert result.metrics["raw_any_feasible"]
    assert result.metrics["best_raw_objective"] is not None
    assert result.metrics["pre_fallback_success"]
    assert result.metrics["best_pre_fallback_objective"] == result.objective
    assert result.metrics["best_pre_fallback_source"] == "raw"


def test_always_available_fallback_participates_in_final_selection() -> None:
    instance = create_toy_instance()
    poor_but_feasible = np.asarray([[0, 1, 1, 0, 1]], dtype=np.int64)
    conditional = solve_from_proposals(
        instance,
        poor_but_feasible,
        config=InferenceConfig(num_samples=1, always_include_fallback=False),
    )
    hybrid = solve_from_proposals(
        instance,
        poor_but_feasible,
        config=InferenceConfig(num_samples=1, always_include_fallback=True),
    )

    assert conditional.source == "raw"
    assert hybrid.metrics["fallback_invoked"]
    assert hybrid.objective <= conditional.objective


def test_fallback_only_baseline_returns_verified_output() -> None:
    instance = create_toy_instance()
    result = solve_fallback_only(instance)

    assert result.success
    assert result.source == "fallback"
    assert verify_placement(instance, result.placement).feasible
    assert result.metrics["raw_feasible_rate"] is None
    assert result.metrics["total_seconds"] >= result.metrics["fallback_seconds"]


def test_greedy_local_baseline_is_independent_and_verified() -> None:
    instance = create_toy_instance()

    construction = construct_greedy_local_placement(instance)
    result = solve_greedy_local(instance)

    assert construction.success
    assert result.success
    assert result.source == "greedy_local"
    assert verify_placement(instance, result.placement).feasible
    assert result.metrics["greedy_assigned_services"] == instance.num_services
    assert result.metrics["optimization_seconds"] >= 0


def test_latency_aware_heuristic_baseline_returns_verified_output() -> None:
    instance = create_toy_instance()

    construction = construct_latency_aware_heuristic_candidates(instance)
    result = solve_latency_aware_heuristic(instance)

    assert construction.success
    assert result.success
    assert result.source == "latency_aware_heuristic"
    assert verify_placement(instance, result.placement).feasible
    assert result.metrics["heuristic_candidate_count"] >= 1
    assert result.metrics["heuristic_selected_strategy"] is not None
    assert result.metrics["heuristic_selected_scoring"] is not None


def test_local_search_baseline_is_verified_and_strictly_nonworsening() -> None:
    instance = create_toy_instance()

    initial = construct_latency_aware_heuristic_candidates(instance)
    trace = run_local_search(instance)
    result = solve_local_search(instance)

    assert trace.success
    assert result.success
    assert result.source == "local_search"
    assert verify_placement(instance, result.placement).feasible
    assert result.metrics["local_search_evaluated_moves"] >= 0
    assert result.metrics["local_search_accepted_moves"] >= 0
    if initial.objective is not None:
        assert result.objective <= initial.objective + 1e-8


def test_time_limited_milp_baseline_reports_solver_evidence() -> None:
    instance = create_toy_instance()

    result = solve_milp_time_limit(instance, time_limit_seconds=5.0, seed=7)

    assert result.success
    assert result.source == "milp_incumbent"
    assert verify_placement(instance, result.placement).feasible
    assert result.metrics["milp_status"] == "OPTIMAL"
    assert result.metrics["milp_optimal"]
    assert result.metrics["milp_gap"] == 0.0


def test_time_limited_milp_maps_large_experiment_seed_deterministically() -> None:
    instance = create_toy_instance()
    seed = 2**63 + 17

    result = solve_milp_time_limit(instance, time_limit_seconds=5.0, seed=seed)

    assert result.success
    assert result.metrics["milp_experiment_seed"] == seed
    assert result.metrics["milp_solver_seed"] == seed % 2_000_000_000
