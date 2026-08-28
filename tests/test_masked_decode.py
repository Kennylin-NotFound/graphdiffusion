import torch
from torch import nn

from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.diffusion import (
    AbsorbingMaskSchedule,
    PartialPlacementState,
    all_masked_state,
)
from gdm_factor_diffusion.graph import build_factor_graph_batch, infer_feature_schema
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    build_residual_candidate_mask,
    decode_partial_batch,
    solve_with_masked_model,
)
from gdm_factor_diffusion.solver import verify_placement


class _OracleModel(nn.Module):
    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target", target)

    def forward(self, batch, state, timestep):
        logits = torch.zeros_like(batch.candidate_mask, dtype=torch.float32)
        target = self.target.to(logits.device).expand(batch.batch_size, -1)
        logits.scatter_(-1, target.unsqueeze(-1), 12.0)
        return logits.masked_fill(~batch.candidate_mask, -torch.inf)


def _toy():
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    target = torch.tensor([[1, 1, 2, 2, 2]])
    return instance, batch, target


def test_residual_mask_enforces_committed_capacity_and_visible_links() -> None:
    _, batch, target = _toy()
    committed = torch.tensor([[True, True, False, False, False]])
    state = PartialPlacementState(target.masked_fill(~committed, -1), committed)
    mask = build_residual_candidate_mask(batch, state)

    assert not mask[committed].any()
    for service in torch.nonzero(~committed[0], as_tuple=False).flatten():
        for device in torch.nonzero(mask[0, service], as_tuple=False).flatten():
            extended_assignment = state.assignment.clone()
            extended_committed = committed.clone()
            extended_assignment[0, service] = device
            extended_committed[0, service] = True
            extended = PartialPlacementState(extended_assignment, extended_committed)
            next_mask = build_residual_candidate_mask(batch, extended)
            assert next_mask.shape == mask.shape


def test_deterministic_and_stochastic_decoders_share_schedule_and_finish() -> None:
    instance, batch, target = _toy()
    model = _OracleModel(target)
    schedule = AbsorbingMaskSchedule(num_steps=4)
    deterministic = decode_partial_batch(
        model, batch, schedule, stochastic=False
    )
    stochastic = decode_partial_batch(
        model,
        batch,
        schedule,
        stochastic=True,
        temperature=0.05,
        generator=torch.Generator().manual_seed(4),
    )

    assert deterministic.completed.all() and stochastic.completed.all()
    assert deterministic.model_forwards == stochastic.model_forwards == 4
    assert torch.equal(deterministic.assignment, target)
    assert verify_placement(instance, stochastic.assignment[0].numpy()).feasible


def test_decoder_terminates_with_explicit_incomplete_state_when_no_choice_exists() -> None:
    _, batch, target = _toy()
    batch.device_capacity.zero_()
    result = decode_partial_batch(
        _OracleModel(target),
        batch,
        AbsorbingMaskSchedule(num_steps=3),
        stochastic=False,
    )

    assert not result.completed.any()
    assert (result.assignment == -1).all()
    assert result.model_forwards == 3


def test_masked_solver_reuses_verifier_and_exact_selection() -> None:
    instance, _, target = _toy()
    result = solve_with_masked_model(
        _OracleModel(target),
        instance,
        AbsorbingMaskSchedule(num_steps=4),
        infer_feature_schema([instance]),
        decode_config=MaskedDecodeConfig(
            num_samples=3,
            sample_batch_size=2,
            stochastic=True,
            temperature=0.05,
        ),
        inference_config=InferenceConfig(
            num_samples=3,
            sample_batch_size=2,
            enable_repair=True,
            enable_fallback=True,
        ),
        generator=torch.Generator().manual_seed(8),
    )

    assert result.success
    assert result.metrics["masked_completed_count"] == 3
    assert result.metrics["final_success"]
    assert verify_placement(instance, result.placement).feasible
