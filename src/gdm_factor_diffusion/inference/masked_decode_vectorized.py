"""Vectorized inference path for the absorbing-MASK conditional decoder."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch
from torch import Tensor, nn

from gdm_factor_diffusion.data.schema import DeploymentInstance
from gdm_factor_diffusion.diffusion import (
    AbsorbingMaskSchedule,
    PartialPlacementState,
    all_masked_state,
    validate_partial_state,
)
from gdm_factor_diffusion.graph import (
    FactorGraphBatch,
    GraphFeatureSchema,
    build_factor_graph_batch,
)

from .masked_decode import (
    MaskedDecodeConfig,
    MaskedProposalBatch,
    PartialDecodeResult,
)
from .solve import InferenceConfig, SolveResult, solve_from_proposals


def build_residual_candidate_mask_vectorized(
    batch: FactorGraphBatch,
    state: PartialPlacementState,
    *,
    tolerance: float = 1e-8,
) -> Tensor:
    """Apply capacity and visible-link constraints without Python edge loops."""

    validate_partial_state(state, batch.candidate_mask, batch.service_mask)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    committed = state.committed_mask
    one_hot = torch.nn.functional.one_hot(
        state.assignment.clamp_min(0),
        num_classes=batch.candidate_mask.shape[-1],
    ).to(dtype=batch.service_demand.dtype)
    one_hot = one_hot * committed.unsqueeze(-1).to(dtype=one_hot.dtype)
    load = torch.einsum("bmd,bmr->bdr", one_hot, batch.service_demand)
    projected = load[:, None, :, :] + batch.service_demand[:, :, None, :]
    capacity_ok = (
        projected <= batch.device_capacity[:, None, :, :] + tolerance
    ).all(dim=-1)
    admissible = batch.candidate_mask & capacity_ok
    admissible &= (batch.service_mask & ~committed).unsqueeze(-1)

    source = batch.dependency_index[:, 0]
    target = batch.dependency_index[:, 1]
    source_visible = torch.gather(committed, 1, source)
    target_visible = torch.gather(committed, 1, target)
    source_device = torch.gather(state.assignment.clamp_min(0), 1, source)
    target_device = torch.gather(state.assignment.clamp_min(0), 1, target)
    device_count = batch.candidate_mask.shape[-1]

    source_selected = batch.pair_admissible.gather(
        2,
        source_device[..., None, None].expand(-1, -1, 1, device_count),
    ).squeeze(2)
    target_selected = batch.pair_admissible.gather(
        3,
        target_device[..., None, None].expand(-1, -1, device_count, 1),
    ).squeeze(3)

    dependency_active = batch.dependency_mask
    restrict_target = dependency_active & source_visible & ~target_visible
    restrict_source = dependency_active & target_visible & ~source_visible
    all_allowed = torch.ones_like(source_selected, dtype=torch.bool)
    target_values = torch.where(
        restrict_target.unsqueeze(-1), source_selected, all_allowed
    )
    source_values = torch.where(
        restrict_source.unsqueeze(-1), target_selected, all_allowed
    )

    # Integer amin is a portable logical-AND reduction for repeated endpoints.
    link_allowed = torch.ones_like(batch.candidate_mask, dtype=torch.int8)
    target_index = target.unsqueeze(-1).expand(-1, -1, device_count)
    source_index = source.unsqueeze(-1).expand(-1, -1, device_count)
    link_allowed.scatter_reduce_(
        1,
        target_index,
        target_values.to(torch.int8),
        reduce="amin",
        include_self=True,
    )
    link_allowed.scatter_reduce_(
        1,
        source_index,
        source_values.to(torch.int8),
        reduce="amin",
        include_self=True,
    )
    return admissible & link_allowed.bool()


def _masked_probability(logits: Tensor, mask: Tensor, temperature: float) -> Tensor:
    row_has_choice = mask.any(dim=-1)
    safe_logits = (logits / temperature).masked_fill(~mask, -torch.inf)
    safe_logits = torch.where(
        row_has_choice.unsqueeze(-1), safe_logits, torch.zeros_like(safe_logits)
    )
    probability = torch.softmax(safe_logits, dim=-1).masked_fill(~mask, 0.0)
    return torch.where(
        row_has_choice.unsqueeze(-1), probability, torch.zeros_like(probability)
    )


def _sample_cdf(probability: Tensor, generator: torch.Generator | None) -> Tensor:
    """Draw one categorical sample per row using a vectorized inverse CDF."""

    uniforms = torch.rand(
        (probability.shape[0], 1),
        dtype=probability.dtype,
        device=probability.device,
        generator=generator,
    )
    cdf = probability.cumsum(dim=-1)
    return (uniforms > cdf).sum(dim=-1).clamp_max(probability.shape[-1] - 1)


@torch.no_grad()
def decode_partial_batch_vectorized(
    model: nn.Module,
    batch: FactorGraphBatch,
    schedule: AbsorbingMaskSchedule,
    *,
    stochastic: bool,
    temperature: float = 1.0,
    tolerance: float = 1e-8,
    generator: torch.Generator | None = None,
) -> PartialDecodeResult:
    """Reverse the MASK schedule with batched commitment updates."""

    if temperature <= 0 or tolerance <= 0:
        raise ValueError("temperature and tolerance must be positive.")
    model.eval()
    state = all_masked_state(batch.candidate_mask, batch.service_mask)
    assignment = state.assignment.clone()
    committed = state.committed_mask.clone()
    commit_probability = torch.zeros_like(
        batch.candidate_mask, dtype=batch.processing_latency.dtype
    )
    service_count = batch.service_mask.sum(dim=1)
    model_forwards = 0

    for timestep in range(schedule.num_steps, 0, -1):
        state = PartialPlacementState(assignment, committed)
        logits = model(batch, state, timestep)
        model_forwards += 1
        visible_fraction = 1.0 - float(
            schedule.mask_probability(timestep - 1).item()
        )
        target_count = torch.ceil(
            service_count.to(dtype=torch.float32) * visible_fraction - 1e-7
        ).to(dtype=torch.long)

        while True:
            needs_commit = committed.sum(dim=1) < target_count
            if not needs_commit.any():
                break
            state = PartialPlacementState(assignment, committed)
            hard_mask = build_residual_candidate_mask_vectorized(
                batch, state, tolerance=tolerance
            )
            probability = _masked_probability(logits, hard_mask, temperature)
            eligible = hard_mask.any(dim=-1)
            confidence = probability.amax(dim=-1).masked_fill(~eligible, -1.0)
            selected_service = confidence.argmax(dim=1)
            active = needs_commit & eligible.any(dim=1)
            if not active.any():
                break
            batch_index = torch.nonzero(active, as_tuple=False).flatten()
            service_index = selected_service[batch_index]
            selected_probability = probability[batch_index, service_index]
            selected_device = (
                _sample_cdf(selected_probability, generator)
                if stochastic
                else selected_probability.argmax(dim=-1)
            )
            assignment[batch_index, service_index] = selected_device
            committed[batch_index, service_index] = True
            commit_probability[batch_index, service_index] = selected_probability

    completed = (committed | ~batch.service_mask).all(dim=1)
    result_state = PartialPlacementState(assignment, committed)
    validate_partial_state(result_state, batch.candidate_mask, batch.service_mask)
    return PartialDecodeResult(
        assignment=assignment,
        completed=completed,
        commit_probability=commit_probability,
        model_forwards=model_forwards,
        transitions=schedule.num_steps,
    )


@torch.no_grad()
def sample_masked_proposals_vectorized(
    model: nn.Module,
    instance: DeploymentInstance,
    schedule: AbsorbingMaskSchedule,
    feature_schema: GraphFeatureSchema,
    *,
    config: MaskedDecodeConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> MaskedProposalBatch:
    settings = config or MaskedDecodeConfig()
    settings.validate()
    target_device = torch.device(device)
    proposals: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    completed: list[np.ndarray] = []
    model_forwards = 0
    start = perf_counter()
    remaining = settings.num_samples
    while remaining:
        chunk = min(settings.sample_batch_size, remaining)
        batch = build_factor_graph_batch(
            [instance] * chunk, feature_schema=feature_schema
        ).to(target_device)
        decoded = decode_partial_batch_vectorized(
            model,
            batch,
            schedule,
            stochastic=settings.stochastic,
            temperature=settings.temperature,
            tolerance=settings.tolerance,
            generator=generator,
        )
        proposals.append(decoded.assignment[:, : instance.num_services].cpu().numpy())
        probabilities.append(
            decoded.commit_probability[
                :, : instance.num_services, : instance.num_devices
            ]
            .cpu()
            .numpy()
        )
        completed.append(decoded.completed.cpu().numpy())
        model_forwards += decoded.model_forwards
        remaining -= chunk
    return MaskedProposalBatch(
        proposals=np.concatenate(proposals, axis=0),
        probabilities=np.concatenate(probabilities, axis=0),
        completed=np.concatenate(completed, axis=0),
        sampling_seconds=perf_counter() - start,
        model_forwards=model_forwards,
    )


def solve_with_masked_model_vectorized(
    model: nn.Module,
    instance: DeploymentInstance,
    schedule: AbsorbingMaskSchedule,
    feature_schema: GraphFeatureSchema,
    *,
    decode_config: MaskedDecodeConfig | None = None,
    inference_config: InferenceConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> SolveResult:
    decoding = decode_config or MaskedDecodeConfig()
    inference = inference_config or InferenceConfig(
        num_samples=decoding.num_samples,
        sample_batch_size=decoding.sample_batch_size,
    )
    if inference.num_samples != decoding.num_samples:
        raise ValueError("Decode and inference proposal counts must agree.")
    sampled = sample_masked_proposals_vectorized(
        model,
        instance,
        schedule,
        feature_schema,
        config=decoding,
        device=device,
        generator=generator,
    )
    result = solve_from_proposals(
        instance,
        sampled.proposals,
        model_probabilities=sampled.probabilities,
        config=inference,
        sampling_seconds=sampled.sampling_seconds,
        proposal_method=(
            "masked_stochastic_vectorized"
            if decoding.stochastic
            else "masked_deterministic_vectorized"
        ),
    )
    result.metrics.update(
        {
            "masked_decode_config": asdict(decoding),
            "masked_completed_count": int(sampled.completed.sum()),
            "masked_completed_rate": float(sampled.completed.mean()),
            "masked_model_forwards": sampled.model_forwards,
            "vectorized_decode": True,
        }
    )
    return result

