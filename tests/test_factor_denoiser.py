import numpy as np
import pytest
import torch
from torch.nn import functional as F

from gdm_factor_diffusion.data import (
    InstanceGenerationSpec,
    build_factor_graph_blueprint,
    create_toy_instance,
    generate_instance,
)
from gdm_factor_diffusion.diffusion import (
    CategoricalSchedule,
    masked_softmax,
    q_sample,
)
from gdm_factor_diffusion.graph import (
    build_dynamic_context,
    build_factor_graph_batch,
    build_hetero_graph,
)
from gdm_factor_diffusion.models import (
    DenoiserConfig,
    DirectPredictorConfig,
    TypedFactorDenoiser,
    TypedFactorDirectPredictor,
)
from gdm_factor_diffusion.solver import evaluate_latency, verify_placement


def _generated(
    *,
    instance_id: str = "factor-test",
    seed: int = 44,
    application_type_ids: tuple[int, ...] = (0, 2),
):
    return generate_instance(
        InstanceGenerationSpec(
            instance_id=instance_id,
            seed=seed,
            partition="train",
            role="train",
            regime="unit",
            size_profile="small",
            num_applications=len(application_type_ids),
            num_devices=4,
            share_probability=0.5,
            compatibility_density=0.6,
            topology_density=0.5,
            capacity_slack=0.4,
            minimum_candidates=2,
            application_type_ids=application_type_ids,
        )
    )


def test_hetero_graph_matches_framework_independent_blueprint() -> None:
    instance = create_toy_instance()
    blueprint = build_factor_graph_blueprint(instance)
    graph = build_hetero_graph(instance)

    assert set(graph.node_types) == {"service", "device", "dependency", "application"}
    assert graph["service"].num_nodes == instance.num_services
    assert graph["device"].num_nodes == instance.num_devices
    for name, expected_index in blueprint.relation_index.items():
        edge_type = tuple(name.split("__"))
        assert np.array_equal(graph[edge_type].edge_index.numpy(), expected_index)
        assert graph[edge_type].edge_attr.shape == (expected_index.shape[1], 1)


def test_batch_adapter_preserves_local_to_flattened_alignment() -> None:
    instances = [create_toy_instance(), _generated().instance]
    batch = build_factor_graph_batch(instances)

    assert batch.batch_size == 2
    assert batch.service_mask.sum().item() == sum(
        instance.num_services for instance in instances
    )
    assert batch.graph["service"].num_nodes == batch.service_mask.sum().item()
    assert batch.graph["device"].num_nodes == sum(
        instance.num_devices for instance in instances
    )
    first_candidates = batch.candidate_mask[
        0, : instances[0].num_services, : instances[0].num_devices
    ]
    assert first_candidates.numpy().tolist() == instances[0].compatibility_mask.tolist()


def test_dynamic_context_matches_exact_toy_processing_load_and_transmission() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    placement = torch.tensor([[0, 1, 1, 0, 1]])
    context = build_dynamic_context(batch, placement)
    evaluation = evaluate_latency(instance, placement[0].numpy())
    verification = verify_placement(instance, placement[0].numpy())

    assert context["service"][:, 0].tolist() == pytest.approx(
        evaluation.processing_latency.tolist()
    )
    assert context["device"].numpy() == pytest.approx(
        verification.capacity_load / instance.device_capacity
    )
    assert context["dependency"][:, 0].tolist() == pytest.approx(
        evaluation.transmission_latency.tolist()
    )
    assert context["dependency"][:, 1].tolist() == pytest.approx([1.0] * 4)

    invalid_link_placement = torch.tensor([[0, 0, 2, 0, 2]])
    invalid_context = build_dynamic_context(batch, invalid_link_placement)
    assert 0.0 in invalid_context["dependency"][:, 1].tolist()


def test_denoiser_forward_backward_and_mask_invariants() -> None:
    generated = _generated()
    instances = [create_toy_instance(), generated.instance]
    batch = build_factor_graph_batch(instances)
    clean_state = torch.full(batch.service_mask.shape, -1, dtype=torch.long)
    clean_state[0, : instances[0].num_services] = torch.tensor([1, 1, 2, 2, 2])
    clean_state[1, : generated.instance.num_services] = torch.from_numpy(
        generated.witness_placement
    )
    generated_target = clean_state[1, : instances[1].num_services].numpy()
    assert verify_placement(instances[1], generated_target).feasible

    schedule = CategoricalSchedule.linear(10, beta_end=0.3)
    timestep = torch.tensor([4, 8])
    noisy_state = q_sample(clean_state, timestep, batch.candidate_mask, schedule)
    model = TypedFactorDenoiser.from_batch(
        batch,
        DenoiserConfig(
            num_diffusion_steps=10,
            hidden_dim=32,
            num_layers=2,
        ),
    )
    logits = model(batch, noisy_state, timestep)
    probability = masked_softmax(logits, batch.candidate_mask, batch.service_mask)

    assert logits.shape == batch.candidate_mask.shape
    assert torch.isneginf(logits[~batch.candidate_mask]).all()
    assert torch.allclose(
        probability.sum(-1)[batch.service_mask],
        torch.ones_like(probability.sum(-1)[batch.service_mask]),
    )
    loss = F.cross_entropy(logits[batch.service_mask], clean_state[batch.service_mask])
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_direct_predictor_forward_backward_and_mask_invariants() -> None:
    generated = _generated()
    instances = [create_toy_instance(), generated.instance]
    batch = build_factor_graph_batch(instances)
    clean_state = torch.full(batch.service_mask.shape, -1, dtype=torch.long)
    clean_state[0, : instances[0].num_services] = torch.tensor([1, 1, 2, 2, 2])
    clean_state[1, : instances[1].num_services] = torch.from_numpy(
        generated.witness_placement
    )
    model = TypedFactorDirectPredictor.from_batch(
        batch,
        DirectPredictorConfig(hidden_dim=32, num_layers=2),
    )
    logits = model(batch)
    probability = masked_softmax(logits, batch.candidate_mask, batch.service_mask)

    assert logits.shape == batch.candidate_mask.shape
    assert torch.isneginf(logits[~batch.candidate_mask]).all()
    assert torch.allclose(
        probability.sum(-1)[batch.service_mask],
        torch.ones_like(probability.sum(-1)[batch.service_mask]),
    )
    loss = F.cross_entropy(logits[batch.service_mask], clean_state[batch.service_mask])
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
def test_denoiser_cuda_smoke() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance]).to("cuda")
    clean_state = torch.tensor([[1, 1, 2, 2, 2]], device="cuda")
    schedule = CategoricalSchedule.linear(10, beta_end=0.3).to("cuda")
    noisy_state = q_sample(clean_state, 7, batch.candidate_mask, schedule)
    model = TypedFactorDenoiser.from_batch(
        batch,
        DenoiserConfig(num_diffusion_steps=10, hidden_dim=32, num_layers=2),
    ).cuda()
    logits = model(batch, noisy_state, 7)
    loss = F.cross_entropy(logits[batch.service_mask], clean_state[batch.service_mask])
    loss.backward()
    assert torch.isfinite(loss)


def test_model_built_on_seen_workflow_accepts_unseen_application_type() -> None:
    train = _generated(
        instance_id="seen-workflow",
        seed=71,
        application_type_ids=(0, 1),
    )
    unseen = _generated(
        instance_id="unseen-workflow",
        seed=72,
        application_type_ids=(5, 5),
    )
    train_batch = build_factor_graph_batch([train.instance])
    unseen_batch = build_factor_graph_batch([unseen.instance])
    model = TypedFactorDenoiser.from_batch(
        train_batch,
        DenoiserConfig(num_diffusion_steps=10, hidden_dim=32, num_layers=2),
    )
    unseen_state = torch.from_numpy(unseen.witness_placement[None, :])
    logits = model(unseen_batch, unseen_state, 5)
    assert logits.shape == unseen_batch.candidate_mask.shape
    assert torch.isfinite(logits[unseen_batch.candidate_mask]).all()
