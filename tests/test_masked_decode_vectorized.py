import torch
from torch import nn

from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.diffusion import AbsorbingMaskSchedule, PartialPlacementState
from gdm_factor_diffusion.graph import build_factor_graph_batch
from gdm_factor_diffusion.inference.masked_decode import (
    build_residual_candidate_mask,
    decode_partial_batch,
)
from gdm_factor_diffusion.inference.masked_decode_vectorized import (
    build_residual_candidate_mask_vectorized,
    decode_partial_batch_vectorized,
)
from gdm_factor_diffusion.solver import verify_placement


class _OracleModel(nn.Module):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target", target)

    def forward(self, batch, state, timestep):
        logits = torch.zeros_like(batch.candidate_mask, dtype=torch.float32)
        target = self.target.to(logits.device).expand(batch.batch_size, -1)
        logits.scatter_(-1, target.unsqueeze(-1), 8.0)
        return logits.masked_fill(~batch.candidate_mask, -torch.inf)


def test_vectorized_residual_mask_matches_legacy_for_repeated_batch() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance] * 4)
    assignment = torch.tensor([[1, 1, -1, -1, -1]]).expand(4, -1).clone()
    committed = assignment >= 0
    state = PartialPlacementState(assignment, committed)
    expected = build_residual_candidate_mask(batch, state)
    actual = build_residual_candidate_mask_vectorized(batch, state)
    assert torch.equal(actual, expected)


def test_vectorized_deterministic_decode_exactly_matches_legacy() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance] * 3)
    target = torch.tensor([[1, 1, 2, 2, 2]])
    model = _OracleModel(target)
    schedule = AbsorbingMaskSchedule(num_steps=4)
    legacy = decode_partial_batch(model, batch, schedule, stochastic=False)
    vectorized = decode_partial_batch_vectorized(
        model, batch, schedule, stochastic=False
    )
    assert torch.equal(vectorized.assignment, legacy.assignment)
    assert torch.equal(vectorized.completed, legacy.completed)
    assert torch.equal(vectorized.commit_probability, legacy.commit_probability)


def test_vectorized_stochastic_decode_replays_and_verifies() -> None:
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance] * 8)
    target = torch.tensor([[1, 1, 2, 2, 2]])
    model = _OracleModel(target)
    schedule = AbsorbingMaskSchedule(num_steps=4)
    first = decode_partial_batch_vectorized(
        model,
        batch,
        schedule,
        stochastic=True,
        generator=torch.Generator().manual_seed(41),
    )
    second = decode_partial_batch_vectorized(
        model,
        batch,
        schedule,
        stochastic=True,
        generator=torch.Generator().manual_seed(41),
    )
    assert torch.equal(first.assignment, second.assignment)
    assert first.completed.all()
    for placement in first.assignment:
        assert verify_placement(instance, placement.numpy()).feasible

