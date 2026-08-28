from pathlib import Path

import numpy as np
import torch
from torch import nn

from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.diffusion import all_masked_state
from gdm_factor_diffusion.graph import build_factor_graph_batch, infer_feature_schema
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    SequentialDecodeConfig,
    decode_sequential_batch,
    solve_with_sequential_model,
)
from gdm_factor_diffusion.models import (
    SequentialPolicyConfig,
    TypedFactorSequentialPolicy,
)
from gdm_factor_diffusion.sequence import service_order, service_order_batch
from gdm_factor_diffusion.solver import verify_placement
from gdm_factor_diffusion.training import (
    SequentialConditionalTrainer,
    SequentialTrainerConfig,
    build_teacher_forced_prefix,
    compute_sequential_objective,
    load_sequential_checkpoint,
    save_sequential_checkpoint,
)


class _SequentialOracle(nn.Module):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target", target)

    def forward(self, batch, state, target_service, step_fraction):
        logits = torch.zeros_like(batch.candidate_mask, dtype=torch.float32)
        target = self.target.to(logits.device).expand(batch.batch_size, -1)
        logits.scatter_(-1, target.unsqueeze(-1), 12.0)
        return logits.masked_fill(~batch.candidate_mask, -torch.inf)


def _toy(device: str = "cpu"):
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance]).to(device)
    target = torch.tensor([[1, 1, 2, 2, 2]], device=device)
    order = service_order_batch([instance]).to(device)
    return instance, batch, target, order


def test_service_order_batch_uses_valid_topological_order() -> None:
    instance = create_toy_instance()
    order = service_order(instance)
    batched = service_order_batch([instance, instance])

    assert sorted(order.tolist()) == list(range(instance.num_services))
    assert batched.shape == (2, instance.num_services)
    assert torch.equal(batched[0], batched[1])


def test_sequential_policy_forward_backward_and_target_context() -> None:
    _, batch, clean, _ = _toy()
    model = TypedFactorSequentialPolicy.from_batch(
        batch,
        SequentialPolicyConfig(hidden_dim=32, num_layers=2),
    )
    partial = all_masked_state(batch.candidate_mask, batch.service_mask)
    first_logits = model(batch, partial, torch.tensor([0]), torch.tensor([0.0]))
    second_logits = model(batch, partial, torch.tensor([1]), torch.tensor([0.0]))
    terms = compute_sequential_objective(
        first_logits,
        clean,
        partial,
        torch.tensor([0]),
        batch,
    )
    terms.total.backward()

    assert first_logits.shape == batch.candidate_mask.shape
    assert torch.isneginf(first_logits[~batch.candidate_mask]).all()
    assert not torch.allclose(
        first_logits[batch.candidate_mask],
        second_logits[batch.candidate_mask],
    )
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_sequential_trainer_and_checkpoint_round_trip(tmp_path: Path) -> None:
    _, batch, clean, order = _toy()
    model = TypedFactorSequentialPolicy.from_batch(
        batch,
        SequentialPolicyConfig(hidden_dim=32, num_layers=2),
    )
    trainer = SequentialConditionalTrainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        SequentialTrainerConfig(gradient_clip_norm=1.0),
    )
    metrics = trainer.train_step(batch, clean, order, torch.tensor([2]))
    saved = next(model.parameters()).detach().clone()
    path = save_sequential_checkpoint(
        tmp_path / "sequential.pt",
        trainer,
        metadata={"tag": "unit"},
    )
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    metadata = load_sequential_checkpoint(path, trainer)

    assert metadata == {"tag": "unit"}
    assert trainer.step == 1
    assert torch.equal(next(model.parameters()), saved)
    assert all(np.isfinite(value) for value in metrics.values())
    assert torch.load(path, weights_only=True)["model_kind"] == "sequential_conditional"


def test_teacher_forced_prefix_predicts_only_next_service() -> None:
    _, batch, clean, order = _toy()
    partial, target_service, step_fraction = build_teacher_forced_prefix(
        clean,
        order,
        torch.tensor([3]),
        batch,
    )

    assert target_service.item() == int(order[0, 3].item())
    assert partial.committed_mask.sum().item() == 3
    assert abs(step_fraction.item() - 3 / 5) < 1e-6
    assert (partial.assignment[~partial.committed_mask] == -1).all()


def test_sequential_decoder_oracle_completes_and_solver_verifies() -> None:
    instance, batch, target, order = _toy()
    model = _SequentialOracle(target)
    decoded = decode_sequential_batch(
        model,
        batch,
        order,
        stochastic=False,
    )
    result = solve_with_sequential_model(
        model,
        instance,
        infer_feature_schema([instance]),
        decode_config=SequentialDecodeConfig(
            num_samples=3,
            sample_batch_size=2,
            stochastic=True,
            temperature=0.05,
        ),
        inference_config=InferenceConfig(
            num_samples=3,
            sample_batch_size=2,
            enable_repair=False,
            enable_fallback=True,
            always_include_fallback=False,
        ),
        generator=torch.Generator().manual_seed(9),
    )

    assert decoded.completed.all()
    assert torch.equal(decoded.assignment, target)
    assert decoded.model_forwards == instance.num_services
    assert result.success
    assert result.metrics["repair_attempts"] == 0
    assert result.metrics["sequential_completed_count"] == 3
    assert verify_placement(instance, result.placement).feasible
