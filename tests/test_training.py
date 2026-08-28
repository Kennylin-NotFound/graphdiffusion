from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.diffusion import CategoricalSchedule, masked_softmax, state_to_one_hot
from gdm_factor_diffusion.graph import build_factor_graph_batch
from gdm_factor_diffusion.models import (
    DenoiserConfig,
    DirectPredictorConfig,
    TypedFactorDenoiser,
    TypedFactorDirectPredictor,
)
from gdm_factor_diffusion.training import (
    ConstrainedValidationConfig,
    DenoiserTrainer,
    LabeledBatch,
    LabeledDeploymentDataset,
    LabeledItem,
    TrainerConfig,
    audit_dataset_freeze,
    capture_random_state,
    capacity_guidance,
    evaluate_constrained_validation,
    link_guidance,
    load_checkpoint,
    make_labeled_collator,
    restore_checkpoint,
    restore_random_state,
    sample_training_batch,
    sample_clean_targets,
    save_checkpoint,
    validation_rank,
)
from gdm_factor_diffusion.solver import (
    SolutionPool,
    compute_energy_distribution,
    enumerate_feasible_placements,
)


DATASET_ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "datasets"
    / "phase1b-smoke"
)


def _toy_pool(instance) -> SolutionPool:
    exhaustive = enumerate_feasible_placements(instance)
    placements = exhaustive.placements[:5]
    latencies = exhaustive.objectives[:5]
    energy, probability = compute_energy_distribution(
        latencies,
        beta=5.0,
        epsilon=1e-12,
    )
    return SolutionPool(
        instance_id=instance.instance_id,
        placements=placements,
        latencies=latencies,
        normalized_energy=energy,
        sampling_probability=probability,
        verified=np.ones(placements.shape[0], dtype=np.bool_),
        metadata={"schema_version": "1.0"},
    )


def test_labeled_dataset_loads_separate_instances_and_pools() -> None:
    dataset = LabeledDeploymentDataset(DATASET_ROOT, partitions=("train",))
    item = dataset[0]
    collate = make_labeled_collator(dataset.feature_schema)
    batch = collate([dataset[0], dataset[1]])

    assert len(dataset) == 4
    assert item.pool.instance_id == item.instance.instance_id
    assert item.pool.placements.shape[1] == item.instance.num_services
    assert batch.factor_graph.batch_size == 2
    assert batch.factor_graph.graph["service"].x.shape[1] == len(
        dataset.feature_schema.service_feature_names
    )


def test_final_dataset_freeze_hashes_are_valid() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "datasets"
        / "phase6c-final-main"
    )
    freeze = audit_dataset_freeze(root)
    dataset = LabeledDeploymentDataset(
        root,
        partitions=("validation",),
        require_freeze=True,
    )

    assert freeze["dataset_instance_count"] == 1024
    assert len(dataset) == 64


def test_target_sampling_modes_return_only_pool_members() -> None:
    instance = create_toy_instance()
    pool = _toy_pool(instance)
    batch = LabeledBatch(
        items=(LabeledItem(instance, pool, "train"),),
        factor_graph=build_factor_graph_batch([instance]),
    )
    best = sample_clean_targets(batch, mode="best")
    energy = sample_clean_targets(
        batch,
        mode="energy",
        generator=torch.Generator().manual_seed(7),
    )
    uniform = sample_clean_targets(
        batch,
        mode="uniform",
        generator=torch.Generator().manual_seed(8),
    )

    assert best.pool_index.item() == 0
    assert np.array_equal(best.state[0].numpy(), pool.placements[0])
    for target in (energy, uniform):
        selected = target.pool_index.item()
        assert np.array_equal(target.state[0].numpy(), pool.placements[selected])
        assert target.latency.item() == pytest.approx(pool.latencies[selected])


def test_guidance_is_zero_for_feasible_one_hot_and_positive_for_violations() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    feasible = torch.tensor([[1, 1, 2, 2, 2]])
    feasible_probability = state_to_one_hot(feasible, batch.candidate_mask)
    assert capacity_guidance(feasible_probability, batch).item() == pytest.approx(0.0)
    assert link_guidance(feasible_probability, batch).item() == pytest.approx(0.0)

    disconnected = torch.tensor([[0, 0, 2, 0, 2]])
    disconnected_probability = state_to_one_hot(disconnected, batch.candidate_mask)
    assert link_guidance(disconnected_probability, batch).item() > 0

    reduced_capacity = instance.device_capacity.copy()
    reduced_capacity[0] = [1.2, 1.2]
    tight_instance = replace(instance, device_capacity=reduced_capacity)
    tight_batch = build_factor_graph_batch([tight_instance])
    uniform_probability = masked_softmax(
        torch.zeros_like(tight_batch.candidate_mask, dtype=torch.float32),
        tight_batch.candidate_mask,
    )
    assert capacity_guidance(uniform_probability, tight_batch).item() > 0


def test_trainer_updates_model_and_checkpoint_round_trip(tmp_path: Path) -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    clean_state = torch.tensor([[1, 1, 2, 2, 2]])
    model = TypedFactorDenoiser.from_batch(
        batch,
        DenoiserConfig(num_diffusion_steps=10, hidden_dim=32, num_layers=2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = DenoiserTrainer(
        model,
        CategoricalSchedule.linear(10, beta_end=0.3),
        optimizer,
        TrainerConfig(capacity_weight=0.1, link_weight=0.1),
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    metrics = trainer.train_step(
        batch,
        clean_state,
        torch.tensor([7]),
        generator=torch.Generator().manual_seed(4),
    )

    assert trainer.step == 1
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
    )

    path = save_checkpoint(tmp_path / "checkpoint.pt", trainer, metadata={"tag": "unit"})
    saved_parameter = next(model.parameters()).detach().clone()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    metadata = load_checkpoint(path, trainer)
    assert metadata == {"tag": "unit"}
    assert trainer.step == 1
    assert torch.equal(next(model.parameters()), saved_parameter)


def test_direct_trainer_updates_model_and_rejects_diffusion_checkpoint(
    tmp_path: Path,
) -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    clean_state = torch.tensor([[1, 1, 2, 2, 2]])
    schedule = CategoricalSchedule.linear(3, beta_end=0.3)
    model = TypedFactorDirectPredictor.from_batch(
        batch,
        DirectPredictorConfig(hidden_dim=16, num_layers=1),
    )
    trainer = DenoiserTrainer(
        model,
        schedule,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        model_kind="direct",
    )
    metrics = trainer.train_step(batch, clean_state)
    path = save_checkpoint(tmp_path / "direct.pt", trainer, metadata={"tag": "direct"})

    assert trainer.step == 1
    assert all(np.isfinite(value) for value in metrics.values())
    assert torch.load(path, weights_only=True)["model_kind"] == "direct"

    diffusion_model = TypedFactorDenoiser.from_batch(
        batch,
        DenoiserConfig(num_diffusion_steps=3, hidden_dim=16, num_layers=1),
    )
    diffusion_trainer = DenoiserTrainer(
        diffusion_model,
        schedule,
        torch.optim.AdamW(diffusion_model.parameters(), lr=1e-3),
    )
    with pytest.raises(ValueError, match="model kinds disagree"):
        restore_checkpoint(path, diffusion_trainer)


def test_runtime_random_state_and_step_batch_resume_exactly() -> None:
    dataset = LabeledDeploymentDataset(DATASET_ROOT, partitions=("train",))
    collate = make_labeled_collator(dataset.feature_schema)
    loader = torch.Generator().manual_seed(31)
    target = torch.Generator().manual_seed(32)
    streams = {"loader": loader, "target": target}

    first_batch = sample_training_batch(
        dataset, collate, batch_size=2, generator=loader
    )
    sample_clean_targets(first_batch, mode="energy", generator=target)
    state = capture_random_state(streams)
    expected_batch = sample_training_batch(
        dataset, collate, batch_size=2, generator=loader
    )
    expected_target = sample_clean_targets(
        expected_batch, mode="energy", generator=target
    )

    restore_random_state(state, streams)
    resumed_batch = sample_training_batch(
        dataset, collate, batch_size=2, generator=loader
    )
    resumed_target = sample_clean_targets(
        resumed_batch, mode="energy", generator=target
    )
    assert [item.instance.instance_id for item in resumed_batch.items] == [
        item.instance.instance_id for item in expected_batch.items
    ]
    assert torch.equal(resumed_target.state, expected_target.state)


def test_checkpoint_exposes_runtime_state(tmp_path: Path) -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    model = TypedFactorDenoiser.from_batch(
        batch,
        DenoiserConfig(num_diffusion_steps=3, hidden_dim=16, num_layers=1),
    )
    trainer = DenoiserTrainer(
        model,
        CategoricalSchedule.linear(3, beta_end=0.3),
        torch.optim.AdamW(model.parameters(), lr=1e-3),
    )
    generator = torch.Generator().manual_seed(44)
    runtime_state = capture_random_state({"train": generator})
    path = save_checkpoint(
        tmp_path / "runtime.pt",
        trainer,
        metadata={"tag": "runtime"},
        runtime_state=runtime_state,
    )
    payload = restore_checkpoint(path, trainer)

    assert payload["metadata"]["tag"] == "runtime"
    assert "torch_cpu" in payload["runtime_state"]
    assert "train" in payload["runtime_state"]["generators"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_random_state_restores_after_cuda_checkpoint_mapping(tmp_path: Path) -> None:
    state = capture_random_state({})
    mapped = {
        **state,
        "torch_cuda": [item.cuda() for item in state["torch_cuda"]],
    }

    restore_random_state(mapped, {})


def test_constrained_validation_and_rank_use_verified_exact_outputs() -> None:
    dataset = LabeledDeploymentDataset(DATASET_ROOT, partitions=("validation",))
    batch = build_factor_graph_batch(
        [dataset[0].instance], feature_schema=dataset.feature_schema
    )
    model = TypedFactorDenoiser.from_batch(
        batch,
        DenoiserConfig(num_diffusion_steps=3, hidden_dim=16, num_layers=1),
    )
    metrics = evaluate_constrained_validation(
        model,
        CategoricalSchedule.linear(3, beta_end=0.3),
        dataset.feature_schema,
        dataset,
        config=ConstrainedValidationConfig(
            num_samples=1,
            sample_batch_size=1,
            reverse_steps=None,
            instance_limit=1,
        ),
        seed=45,
    )

    assert metrics["instances"] == 1
    assert metrics["verified_rate"] == pytest.approx(1.0)
    assert metrics["mean_gap_to_pool_best"] is not None
    assert validation_rank(metrics, 0.5) < validation_rank(
        {**metrics, "verified_rate": 0.0}, 0.1
    )


def test_direct_constrained_validation_uses_shared_hard_pipeline() -> None:
    dataset = LabeledDeploymentDataset(DATASET_ROOT, partitions=("validation",))
    batch = build_factor_graph_batch(
        [dataset[0].instance], feature_schema=dataset.feature_schema
    )
    model = TypedFactorDirectPredictor.from_batch(
        batch,
        DirectPredictorConfig(hidden_dim=16, num_layers=1),
    )
    metrics = evaluate_constrained_validation(
        model,
        CategoricalSchedule.linear(3, beta_end=0.3),
        dataset.feature_schema,
        dataset,
        config=ConstrainedValidationConfig(
            num_samples=1,
            sample_batch_size=1,
            reverse_steps=1,
            instance_limit=1,
        ),
        seed=46,
        model_kind="direct",
    )

    assert metrics["instances"] == 1
    assert metrics["verified_rate"] == pytest.approx(1.0)
    assert metrics["mean_gap_to_pool_best"] is not None
