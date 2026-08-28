import pytest
import torch

from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.diffusion import (
    AbsorbingMaskSchedule,
    PartialPlacementState,
    all_masked_state,
    corrupt_with_absorbing_mask,
    hidden_service_mask,
    validate_partial_state,
)
from gdm_factor_diffusion.graph import build_factor_graph_batch, build_partial_context


def _toy():
    instance = create_toy_instance()
    batch = build_factor_graph_batch([instance])
    clean = torch.tensor([[1, 1, 2, 2, 2]])
    return instance, batch, clean


def test_partial_state_separates_mask_from_padding_and_compatibility() -> None:
    _, batch, clean = _toy()
    committed = torch.tensor([[True, False, True, False, True]])
    assignment = clean.clone().masked_fill(~committed, -1)
    state = PartialPlacementState(assignment, committed)

    validate_partial_state(state, batch.candidate_mask, batch.service_mask)
    assert hidden_service_mask(state, batch.candidate_mask).sum().item() == 2

    with pytest.raises(ValueError, match="-1"):
        validate_partial_state(
            PartialPlacementState(clean, committed),
            batch.candidate_mask,
            batch.service_mask,
        )


def test_absorbing_mask_endpoints_monotonicity_and_replay() -> None:
    _, batch, clean = _toy()
    schedule = AbsorbingMaskSchedule(num_steps=4)
    at_zero = corrupt_with_absorbing_mask(
        clean, 0, batch.candidate_mask, schedule, generator=torch.Generator().manual_seed(1)
    )
    at_end = corrupt_with_absorbing_mask(
        clean, 4, batch.candidate_mask, schedule, generator=torch.Generator().manual_seed(1)
    )
    first = corrupt_with_absorbing_mask(
        clean, 2, batch.candidate_mask, schedule, generator=torch.Generator().manual_seed(7)
    )
    second = corrupt_with_absorbing_mask(
        clean, 2, batch.candidate_mask, schedule, generator=torch.Generator().manual_seed(7)
    )

    assert at_zero.committed_mask.equal(batch.service_mask)
    assert at_end.committed_mask.sum().item() == 0
    assert torch.equal(first.assignment, second.assignment)
    assert torch.equal(first.committed_mask, second.committed_mask)
    probabilities = schedule.mask_probability(torch.arange(5))
    assert torch.all(probabilities[1:] >= probabilities[:-1])


def test_positive_timestep_always_hides_an_active_service() -> None:
    _, batch, clean = _toy()
    schedule = AbsorbingMaskSchedule(num_steps=100)
    partial = corrupt_with_absorbing_mask(
        clean,
        1,
        batch.candidate_mask,
        schedule,
        generator=torch.Generator().manual_seed(99),
    )
    assert hidden_service_mask(partial, batch.candidate_mask).any()


def test_partial_context_counts_only_committed_loads_and_visible_links() -> None:
    instance, batch, clean = _toy()
    committed = torch.tensor([[True, True, False, False, False]])
    state = PartialPlacementState(clean.masked_fill(~committed, -1), committed)
    context = build_partial_context(batch, state)

    expected_load = torch.zeros_like(batch.device_capacity)
    for service in (0, 1):
        device = int(clean[0, service])
        expected_load[0, device] += batch.service_demand[0, service]
    assert torch.allclose(context["resource_load"], expected_load)
    assert context["service_dense"][0, :, 0].sum().item() == 2

    source = instance.dependency_index[0]
    target = instance.dependency_index[1]
    both = [bool(committed[0, u] and committed[0, v]) for u, v in zip(source, target)]
    visible_admissible = context["dependency_dense"][0, :, 3]
    assert (visible_admissible > 0).tolist() == both


def test_all_masked_state_is_valid() -> None:
    _, batch, _ = _toy()
    state = all_masked_state(batch.candidate_mask, batch.service_mask)
    validate_partial_state(state, batch.candidate_mask, batch.service_mask)
    assert (state.assignment == -1).all()
    assert not state.committed_mask.any()
