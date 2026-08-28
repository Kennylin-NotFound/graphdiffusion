"""Autoregressive proposal generation for sequential conditional policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch
from torch import Tensor, nn

from gdm_factor_diffusion.data.schema import DeploymentInstance
from gdm_factor_diffusion.diffusion import PartialPlacementState, all_masked_state
from gdm_factor_diffusion.graph import (
    FactorGraphBatch,
    GraphFeatureSchema,
    build_factor_graph_batch,
)
from gdm_factor_diffusion.sequence import service_order_batch

from .masked_decode import build_residual_candidate_mask
from .solve import InferenceConfig, SolveResult, solve_from_proposals


@dataclass(frozen=True, slots=True)
class SequentialDecodeConfig:
    num_samples: int = 4
    sample_batch_size: int = 4
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
class SequentialDecodeResult:
    assignment: Tensor
    completed: Tensor
    commit_probability: Tensor
    model_forwards: int
    decisions: int


@dataclass(frozen=True, slots=True)
class SequentialProposalBatch:
    proposals: np.ndarray
    probabilities: np.ndarray
    completed: np.ndarray
    sampling_seconds: float
    model_forwards: int


def _masked_probability(logits: Tensor, mask: Tensor, temperature: float) -> Tensor:
    row_has_choice = mask.any(dim=-1)
    safe_logits = (logits / temperature).masked_fill(~mask, -torch.inf)
    safe_logits = torch.where(
        row_has_choice.unsqueeze(-1),
        safe_logits,
        torch.zeros_like(safe_logits),
    )
    probability = torch.softmax(safe_logits, dim=-1).masked_fill(~mask, 0.0)
    return torch.where(
        row_has_choice.unsqueeze(-1),
        probability,
        torch.zeros_like(probability),
    )


@torch.no_grad()
def decode_sequential_batch(
    model: nn.Module,
    batch: FactorGraphBatch,
    order: Tensor,
    *,
    stochastic: bool,
    temperature: float = 1.0,
    tolerance: float = 1e-8,
    generator: torch.Generator | None = None,
) -> SequentialDecodeResult:
    """Complete services in a fixed order using residual hard masks."""

    if temperature <= 0 or tolerance <= 0:
        raise ValueError("temperature and tolerance must be positive.")
    if order.shape != batch.service_mask.shape:
        raise ValueError("order must have shape [B, M].")
    model.eval()
    normalized_order = order.to(device=batch.candidate_mask.device, dtype=torch.long)
    state = all_masked_state(batch.candidate_mask, batch.service_mask)
    assignment = state.assignment.clone()
    committed = state.committed_mask.clone()
    completed = torch.ones(
        batch.batch_size,
        dtype=torch.bool,
        device=batch.candidate_mask.device,
    )
    commit_probability = torch.zeros_like(
        batch.candidate_mask,
        dtype=batch.processing_latency.dtype,
    )
    service_count = batch.service_mask.sum(dim=1)
    row = torch.arange(batch.batch_size, device=batch.candidate_mask.device)
    model_forwards = 0
    decisions = 0

    for position in range(batch.service_mask.shape[1]):
        active = (position < service_count) & completed
        if not active.any():
            continue
        state = PartialPlacementState(assignment, committed)
        target_service = normalized_order[:, position].clamp_min(0)
        step_fraction = torch.full(
            (batch.batch_size,),
            float(position),
            dtype=batch.processing_latency.dtype,
            device=batch.candidate_mask.device,
        ) / service_count.to(dtype=batch.processing_latency.dtype).clamp_min(1)
        logits = model(batch, state, target_service, step_fraction)
        model_forwards += 1
        hard_mask = build_residual_candidate_mask(
            batch,
            state,
            tolerance=tolerance,
        )
        probability = _masked_probability(logits, hard_mask, temperature)

        for batch_index in torch.nonzero(active, as_tuple=False).flatten():
            service = int(target_service[batch_index].item())
            choices = hard_mask[batch_index, service]
            if not bool(choices.any().item()):
                completed[batch_index] = False
                continue
            service_probability = probability[batch_index, service]
            if stochastic:
                device = int(
                    torch.multinomial(
                        service_probability,
                        num_samples=1,
                        replacement=True,
                        generator=generator,
                    ).item()
                )
            else:
                device = int(service_probability.argmax().item())
            assignment[batch_index, service] = device
            committed[batch_index, service] = True
            commit_probability[batch_index, service] = service_probability
            decisions += 1

    completed = completed & (committed | ~batch.service_mask).all(dim=1)
    return SequentialDecodeResult(
        assignment=assignment,
        completed=completed,
        commit_probability=commit_probability,
        model_forwards=model_forwards,
        decisions=decisions,
    )


@torch.no_grad()
def sample_sequential_proposals(
    model: nn.Module,
    instance: DeploymentInstance,
    feature_schema: GraphFeatureSchema,
    *,
    config: SequentialDecodeConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> SequentialProposalBatch:
    """Generate sequential conditional proposals for one deployment instance."""

    settings = config or SequentialDecodeConfig()
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
            [instance] * chunk,
            feature_schema=feature_schema,
        ).to(target_device)
        order = service_order_batch(
            [instance] * chunk,
            max_services=batch.candidate_mask.shape[1],
        ).to(target_device)
        decoded = decode_sequential_batch(
            model,
            batch,
            order,
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
    return SequentialProposalBatch(
        proposals=np.concatenate(proposals, axis=0),
        probabilities=np.concatenate(probabilities, axis=0),
        completed=np.concatenate(completed, axis=0),
        sampling_seconds=perf_counter() - start,
        model_forwards=model_forwards,
    )


def solve_with_sequential_model(
    model: nn.Module,
    instance: DeploymentInstance,
    feature_schema: GraphFeatureSchema,
    *,
    decode_config: SequentialDecodeConfig | None = None,
    inference_config: InferenceConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> SolveResult:
    """Generate sequential proposals, then reuse hard verification and selection."""

    decoding = decode_config or SequentialDecodeConfig()
    inference = inference_config or InferenceConfig(
        num_samples=decoding.num_samples,
        sample_batch_size=decoding.sample_batch_size,
    )
    if inference.num_samples != decoding.num_samples:
        raise ValueError("Decode and inference proposal counts must agree.")
    sampled = sample_sequential_proposals(
        model,
        instance,
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
            "sequential_stochastic" if decoding.stochastic else "sequential_deterministic"
        ),
    )
    result.metrics.update(
        {
            "sequential_decode_config": asdict(decoding),
            "sequential_completed_count": int(sampled.completed.sum()),
            "sequential_completed_rate": float(sampled.completed.mean()),
            "sequential_model_forwards": sampled.model_forwards,
        }
    )
    return result
