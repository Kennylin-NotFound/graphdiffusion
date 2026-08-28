"""Hard-masked deterministic and stochastic completion of partial placements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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

from .solve import InferenceConfig, SolveResult, solve_from_proposals


@dataclass(frozen=True, slots=True)
class MaskedDecodeConfig:
    num_samples: int = 8
    sample_batch_size: int = 8
    stochastic: bool = True
    temperature: float = 1.0
    tolerance: float = 1e-8

    def validate(self) -> None:
        if self.num_samples < 1 or self.sample_batch_size < 1:
            raise ValueError("num_samples and sample_batch_size must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive.")


@dataclass(frozen=True, slots=True)
class PartialDecodeResult:
    assignment: Tensor
    completed: Tensor
    commit_probability: Tensor
    model_forwards: int
    transitions: int


@dataclass(frozen=True, slots=True)
class MaskedProposalBatch:
    proposals: np.ndarray
    probabilities: np.ndarray
    completed: np.ndarray
    sampling_seconds: float
    model_forwards: int


def build_residual_candidate_mask(
    batch: FactorGraphBatch,
    state: PartialPlacementState,
    *,
    tolerance: float = 1e-8,
) -> Tensor:
    """Mask choices violating compatibility, residual capacity, or visible links."""

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

    for batch_index in range(batch.batch_size):
        for edge_index in torch.nonzero(
            batch.dependency_mask[batch_index], as_tuple=False
        ).flatten():
            source = int(batch.dependency_index[batch_index, 0, edge_index])
            target = int(batch.dependency_index[batch_index, 1, edge_index])
            source_visible = bool(committed[batch_index, source])
            target_visible = bool(committed[batch_index, target])
            if source_visible and not target_visible:
                source_device = int(state.assignment[batch_index, source])
                admissible[batch_index, target] &= batch.pair_admissible[
                    batch_index, edge_index, source_device, :
                ]
            elif target_visible and not source_visible:
                target_device = int(state.assignment[batch_index, target])
                admissible[batch_index, source] &= batch.pair_admissible[
                    batch_index, edge_index, :, target_device
                ]
    return admissible


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


@torch.no_grad()
def decode_partial_batch(
    model: nn.Module,
    batch: FactorGraphBatch,
    schedule: AbsorbingMaskSchedule,
    *,
    stochastic: bool,
    temperature: float = 1.0,
    tolerance: float = 1e-8,
    generator: torch.Generator | None = None,
) -> PartialDecodeResult:
    """Reverse the MASK schedule with bounded, hard-feasible commitments."""

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
            current_count = committed.sum(dim=1)
            needs_commit = current_count < target_count
            if not needs_commit.any():
                break
            state = PartialPlacementState(assignment, committed)
            hard_mask = build_residual_candidate_mask(
                batch, state, tolerance=tolerance
            )
            probability = _masked_probability(logits, hard_mask, temperature)
            progress = False
            for batch_index in torch.nonzero(needs_commit, as_tuple=False).flatten():
                eligible = hard_mask[batch_index].any(dim=-1)
                if not eligible.any():
                    continue
                confidence = probability[batch_index].amax(dim=-1)
                confidence = confidence.masked_fill(~eligible, -1.0)
                service = int(confidence.argmax())
                service_probability = probability[batch_index, service]
                if stochastic:
                    device = int(
                        torch.multinomial(
                            service_probability,
                            num_samples=1,
                            replacement=True,
                            generator=generator,
                        )
                    )
                else:
                    device = int(service_probability.argmax())
                assignment[batch_index, service] = device
                committed[batch_index, service] = True
                commit_probability[batch_index, service] = service_probability
                progress = True
            if not progress:
                break

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
def sample_masked_proposals(
    model: nn.Module,
    instance: DeploymentInstance,
    schedule: AbsorbingMaskSchedule,
    feature_schema: GraphFeatureSchema,
    *,
    config: MaskedDecodeConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> MaskedProposalBatch:
    """Generate a bounded set of deterministic or stochastic partial completions."""

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
        decoded = decode_partial_batch(
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


def solve_with_masked_model(
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
    """Generate partial completions, then reuse verification and exact selection."""

    decoding = decode_config or MaskedDecodeConfig()
    inference = inference_config or InferenceConfig(
        num_samples=decoding.num_samples,
        sample_batch_size=decoding.sample_batch_size,
    )
    if inference.num_samples != decoding.num_samples:
        raise ValueError("Decode and inference proposal counts must agree.")
    sampled = sample_masked_proposals(
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
            "masked_stochastic" if decoding.stochastic else "masked_deterministic"
        ),
    )
    result.metrics.update(
        {
            "masked_decode_config": asdict(decoding),
            "masked_completed_count": int(sampled.completed.sum()),
            "masked_completed_rate": float(sampled.completed.mean()),
            "masked_model_forwards": sampled.model_forwards,
        }
    )
    return result
