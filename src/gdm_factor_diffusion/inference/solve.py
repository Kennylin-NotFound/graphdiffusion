"""Proposal verification, bounded postprocessing, and exact selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from gdm_factor_diffusion.data.schema import DeploymentInstance
from gdm_factor_diffusion.diffusion import (
    CategoricalSchedule,
    build_reverse_timestep_grid,
    masked_softmax,
    reverse_sample_loop,
    sample_categorical,
    sample_prior,
)
from gdm_factor_diffusion.graph import GraphFeatureSchema, build_factor_graph_batch
from gdm_factor_diffusion.solver import evaluate_latency, verify_placement

from .fallback import ConstructiveFallbackConfig, construct_feasible_placement
from .recovery import ProposalRecoveryConfig, recover_from_proposal
from .repair import RepairConfig, repair_placement, violation_score


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    num_samples: int = 8
    sample_batch_size: int = 8
    repair_max_moves: int = 10
    fallback_max_search_nodes: int = 100_000
    enable_repair: bool = True
    enable_recovery: bool = False
    enable_fallback: bool = True
    always_include_fallback: bool = False
    reverse_steps: int | None = None
    repair_candidate_limit: int | None = None
    recovery_candidate_limit: int | None = 4
    recovery_max_released_services: int = 4

    def validate(self) -> None:
        if self.num_samples < 1 or self.sample_batch_size < 1:
            raise ValueError("num_samples and sample_batch_size must be positive.")
        if self.repair_max_moves < 0:
            raise ValueError("repair_max_moves must be nonnegative.")
        if self.fallback_max_search_nodes < 1:
            raise ValueError("fallback_max_search_nodes must be positive.")
        if self.reverse_steps is not None and self.reverse_steps < 1:
            raise ValueError("reverse_steps must be positive when provided.")
        if self.repair_candidate_limit is not None and self.repair_candidate_limit < 1:
            raise ValueError("repair_candidate_limit must be positive when provided.")
        if (
            self.recovery_candidate_limit is not None
            and self.recovery_candidate_limit < 1
        ):
            raise ValueError("recovery_candidate_limit must be positive when provided.")
        if self.recovery_max_released_services < 0:
            raise ValueError("recovery_max_released_services must be nonnegative.")
        if self.enable_recovery and self.enable_repair:
            raise ValueError("Proposal recovery cannot be combined with legacy repair.")
        if self.enable_recovery and self.enable_fallback:
            raise ValueError("Proposal recovery cannot be combined with legacy fallback.")


@dataclass(frozen=True, slots=True)
class VerifiedCandidate:
    placement: np.ndarray
    objective: float
    source: str
    proposal_index: int | None
    repair_moves: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "placement": self.placement.tolist(),
            "objective": self.objective,
            "source": self.source,
            "proposal_index": self.proposal_index,
            "repair_moves": self.repair_moves,
        }


@dataclass(frozen=True, slots=True)
class SolveResult:
    instance_id: str
    placement: np.ndarray | None
    objective: float | None
    source: str
    verified_candidates: tuple[VerifiedCandidate, ...]
    metrics: dict[str, Any]

    @property
    def success(self) -> bool:
        return self.placement is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "placement": None if self.placement is None else self.placement.tolist(),
            "objective": self.objective,
            "source": self.source,
            "success": self.success,
            "verified_candidates": [
                candidate.to_dict() for candidate in self.verified_candidates
            ],
            "metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryProposalBatch:
    """Final reverse samples and clean predictions observed along the same chain."""

    final_proposals: np.ndarray
    final_probabilities: np.ndarray
    clean_proposals: dict[int, np.ndarray]
    clean_probabilities: dict[int, np.ndarray]
    transition_timesteps: dict[int, tuple[int, int]]
    sampling_seconds: float


@dataclass(frozen=True, slots=True)
class TrajectoryCandidateSet:
    """Deduplicated final and intermediate candidates for shared post-processing."""

    proposals: np.ndarray
    probabilities: np.ndarray
    sources: tuple[str, ...]
    candidates_before_deduplication: int
    preparation_seconds: float


@torch.no_grad()
def sample_reverse_proposals(
    model: nn.Module,
    instance: DeploymentInstance,
    schedule: CategoricalSchedule,
    feature_schema: GraphFeatureSchema,
    *,
    config: InferenceConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate learned categorical proposals and final clean-state probabilities."""

    settings = config or InferenceConfig()
    settings.validate()
    target_device = torch.device(device)
    model.eval()
    proposals: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    start = perf_counter()
    reverse_grid = build_reverse_timestep_grid(
        schedule.num_steps,
        schedule.num_steps if settings.reverse_steps is None else settings.reverse_steps,
    )
    remaining = settings.num_samples
    while remaining:
        chunk = min(settings.sample_batch_size, remaining)
        batch = build_factor_graph_batch(
            [instance] * chunk,
            feature_schema=feature_schema,
        ).to(target_device)
        state = reverse_sample_loop(
            lambda noisy, timestep: model(batch, noisy, timestep),
            batch.candidate_mask,
            schedule,
            batch.service_mask,
            timesteps=reverse_grid,
            generator=generator,
        )
        timestep = torch.ones(chunk, dtype=torch.long, device=target_device)
        logits = model(batch, state, timestep)
        probability = masked_softmax(
            logits, batch.candidate_mask, batch.service_mask
        )
        proposals.append(state[:, : instance.num_services].cpu().numpy())
        probabilities.append(
            probability[:, : instance.num_services, : instance.num_devices]
            .cpu()
            .numpy()
        )
        remaining -= chunk
    return (
        np.concatenate(proposals, axis=0),
        np.concatenate(probabilities, axis=0),
        perf_counter() - start,
    )


@torch.no_grad()
def sample_reverse_trajectory_proposals(
    model: nn.Module,
    instance: DeploymentInstance,
    schedule: CategoricalSchedule,
    feature_schema: GraphFeatureSchema,
    *,
    anchor_indices: tuple[int, ...],
    config: InferenceConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> TrajectoryProposalBatch:
    """Collect clean argmax candidates without adding denoiser forward passes."""

    settings = config or InferenceConfig()
    settings.validate()
    target_device = torch.device(device)
    model.eval()
    reverse_steps = (
        schedule.num_steps if settings.reverse_steps is None else settings.reverse_steps
    )
    anchors = tuple(sorted(set(int(index) for index in anchor_indices)))
    if any(index < 0 or index >= reverse_steps for index in anchors):
        raise ValueError("anchor_indices must lie in [0, reverse_steps).")

    final_proposals: list[np.ndarray] = []
    final_probabilities: list[np.ndarray] = []
    clean_proposals: dict[int, list[np.ndarray]] = {index: [] for index in anchors}
    clean_probabilities: dict[int, list[np.ndarray]] = {
        index: [] for index in anchors
    }
    transition_timesteps: dict[int, tuple[int, int]] = {}
    start = perf_counter()
    reverse_grid = build_reverse_timestep_grid(schedule.num_steps, reverse_steps)
    remaining = settings.num_samples
    while remaining:
        chunk = min(settings.sample_batch_size, remaining)
        batch = build_factor_graph_batch(
            [instance] * chunk,
            feature_schema=feature_schema,
        ).to(target_device)
        callback_index = 0

        def observe(
            timestep: int,
            previous_timestep: int,
            _noisy_state: Tensor,
            clean_logits: Tensor,
            _previous_state: Tensor,
        ) -> None:
            nonlocal callback_index
            index = callback_index
            callback_index += 1
            if index not in clean_proposals:
                return
            probability = masked_softmax(
                clean_logits,
                batch.candidate_mask,
                batch.service_mask,
            )
            clean_state = probability.argmax(dim=-1)
            clean_proposals[index].append(
                clean_state[:, : instance.num_services].cpu().numpy()
            )
            clean_probabilities[index].append(
                probability[:, : instance.num_services, : instance.num_devices]
                .cpu()
                .numpy()
            )
            transition_timesteps[index] = (timestep, previous_timestep)

        state = reverse_sample_loop(
            lambda noisy, timestep: model(batch, noisy, timestep),
            batch.candidate_mask,
            schedule,
            batch.service_mask,
            timesteps=reverse_grid,
            generator=generator,
            step_callback=observe,
        )
        timestep = torch.ones(chunk, dtype=torch.long, device=target_device)
        logits = model(batch, state, timestep)
        probability = masked_softmax(
            logits,
            batch.candidate_mask,
            batch.service_mask,
        )
        final_proposals.append(state[:, : instance.num_services].cpu().numpy())
        final_probabilities.append(
            probability[:, : instance.num_services, : instance.num_devices]
            .cpu()
            .numpy()
        )
        remaining -= chunk

    return TrajectoryProposalBatch(
        final_proposals=np.concatenate(final_proposals, axis=0),
        final_probabilities=np.concatenate(final_probabilities, axis=0),
        clean_proposals={
            index: np.concatenate(values, axis=0)
            for index, values in clean_proposals.items()
        },
        clean_probabilities={
            index: np.concatenate(values, axis=0)
            for index, values in clean_probabilities.items()
        },
        transition_timesteps=transition_timesteps,
        sampling_seconds=perf_counter() - start,
    )


def build_trajectory_candidate_set(
    trajectory: TrajectoryProposalBatch,
    *,
    anchor_indices: tuple[int, ...],
) -> TrajectoryCandidateSet:
    """Combine final and selected clean candidates while preserving first origin."""

    start = perf_counter()
    anchors = tuple(int(index) for index in anchor_indices)
    missing = set(anchors) - set(trajectory.clean_proposals)
    if missing:
        raise ValueError(f"Trajectory did not collect anchors: {sorted(missing)}")
    placements = [trajectory.final_proposals]
    probabilities = [trajectory.final_probabilities]
    sources = ["final"] * trajectory.final_proposals.shape[0]
    for index in anchors:
        placements.append(trajectory.clean_proposals[index])
        probabilities.append(trajectory.clean_probabilities[index])
        sources.extend([f"clean_t{index}"] * trajectory.clean_proposals[index].shape[0])
    stacked = np.concatenate(placements, axis=0)
    stacked_probabilities = np.concatenate(probabilities, axis=0)
    keep: list[int] = []
    seen: set[tuple[int, ...]] = set()
    for index, placement in enumerate(stacked):
        key = tuple(int(value) for value in placement)
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)
    return TrajectoryCandidateSet(
        proposals=stacked[keep],
        probabilities=stacked_probabilities[keep],
        sources=tuple(sources[index] for index in keep),
        candidates_before_deduplication=int(stacked.shape[0]),
        preparation_seconds=perf_counter() - start,
    )


def sample_random_proposals(
    instance: DeploymentInstance,
    *,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> np.ndarray:
    """Sample the same factorized compatible-device prior used at diffusion time T."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    candidate_mask = torch.from_numpy(instance.compatibility_mask).unsqueeze(0)
    candidate_mask = candidate_mask.expand(num_samples, -1, -1)
    return sample_prior(candidate_mask, generator=generator).numpy()


@torch.no_grad()
def sample_direct_proposals(
    model: nn.Module,
    instance: DeploymentInstance,
    feature_schema: GraphFeatureSchema,
    *,
    config: InferenceConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample factorized proposals from one-pass clean placement logits."""

    settings = config or InferenceConfig()
    settings.validate()
    target_device = torch.device(device)
    model.eval()
    proposals: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    start = perf_counter()
    remaining = settings.num_samples
    while remaining:
        chunk = min(settings.sample_batch_size, remaining)
        batch = build_factor_graph_batch(
            [instance] * chunk,
            feature_schema=feature_schema,
        ).to(target_device)
        logits = model(batch)
        probability = masked_softmax(
            logits, batch.candidate_mask, batch.service_mask
        )
        state = sample_categorical(
            probability,
            batch.candidate_mask,
            batch.service_mask,
            generator=generator,
        )
        proposals.append(state[:, : instance.num_services].cpu().numpy())
        probabilities.append(
            probability[:, : instance.num_services, : instance.num_devices]
            .cpu()
            .numpy()
        )
        remaining -= chunk
    return (
        np.concatenate(proposals, axis=0),
        np.concatenate(probabilities, axis=0),
        perf_counter() - start,
    )


def _add_verified_candidate(
    instance: DeploymentInstance,
    candidates: dict[tuple[int, ...], VerifiedCandidate],
    placement: np.ndarray,
    *,
    source: str,
    proposal_index: int | None,
    repair_moves: int,
) -> tuple[float, float]:
    verification_start = perf_counter()
    report = verify_placement(instance, placement)
    verification_seconds = perf_counter() - verification_start
    if not report.feasible:
        return verification_seconds, 0.0
    evaluation_start = perf_counter()
    objective = evaluate_latency(instance, report.placement).objective
    evaluation_seconds = perf_counter() - evaluation_start
    key = tuple(int(value) for value in report.placement)
    candidate = VerifiedCandidate(
        placement=report.placement.copy(),
        objective=objective,
        source=source,
        proposal_index=proposal_index,
        repair_moves=repair_moves,
    )
    previous = candidates.get(key)
    if previous is None or (
        candidate.objective,
        candidate.source,
        candidate.proposal_index if candidate.proposal_index is not None else -1,
    ) < (
        previous.objective,
        previous.source,
        previous.proposal_index if previous.proposal_index is not None else -1,
    ):
        candidates[key] = candidate
    return verification_seconds, evaluation_seconds


def _proposal_diagnostics(proposals: np.ndarray) -> dict[str, float | int]:
    """Summarize categorical proposal coverage without quadratic storage."""

    count, services = proposals.shape
    unique_count = int(np.unique(proposals, axis=0).shape[0])
    if count < 2:
        pairwise_hamming = 0.0
    else:
        total_pairs = count * (count - 1) // 2
        unequal_pairs = 0
        for service in range(services):
            _, frequencies = np.unique(proposals[:, service], return_counts=True)
            equal_pairs = int(np.sum(frequencies * (frequencies - 1) // 2))
            unequal_pairs += total_pairs - equal_pairs
        pairwise_hamming = unequal_pairs / (total_pairs * services)
    return {
        "raw_unique_count": unique_count,
        "raw_unique_rate": unique_count / count,
        "raw_pairwise_hamming": float(pairwise_hamming),
    }


def solve_from_proposals(
    instance: DeploymentInstance,
    proposals: np.ndarray,
    *,
    model_probabilities: np.ndarray | None = None,
    config: InferenceConfig | None = None,
    sampling_seconds: float = 0.0,
    proposal_preparation_seconds: float = 0.0,
    proposal_method: str = "external",
) -> SolveResult:
    """Verify, optionally recover, and rank categorical proposals."""

    settings = config or InferenceConfig(num_samples=len(proposals))
    settings.validate()
    raw = np.asarray(proposals, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != instance.num_services or raw.shape[0] < 1:
        raise ValueError("proposals must have shape [K, M] with K >= 1.")
    probabilities = None
    if model_probabilities is not None:
        probabilities = np.asarray(model_probabilities, dtype=np.float64)
        expected = (raw.shape[0], instance.num_services, instance.num_devices)
        if probabilities.shape != expected:
            raise ValueError(f"model_probabilities must have shape {expected}.")

    post_sampling_start = perf_counter()
    proposal_diagnostics = _proposal_diagnostics(raw)
    candidates: dict[tuple[int, ...], VerifiedCandidate] = {}
    raw_feasible = 0
    raw_capacity_violations = 0
    raw_link_violations = 0
    repair_attempts = 0
    repair_successes = 0
    total_repair_moves = 0
    recovery_invoked = False
    recovery_attempts = 0
    recovery_successes = 0
    recovery_released_services = 0
    recovery_completion_steps = 0
    recovery_failure_reasons: dict[str, int] = {}
    candidate_verification_seconds = 0.0
    exact_evaluation_seconds = 0.0
    verification_start = perf_counter()
    reports = [verify_placement(instance, placement) for placement in raw]
    verification_seconds = perf_counter() - verification_start
    for index, report in enumerate(reports):
        raw_feasible += int(report.feasible)
        raw_capacity_violations += int(not report.capacity_valid)
        raw_link_violations += int(not report.direct_link_valid)
        if report.feasible:
            candidate_verification, exact_evaluation = _add_verified_candidate(
                instance,
                candidates,
                report.placement,
                source="raw",
                proposal_index=index,
                repair_moves=0,
            )
            candidate_verification_seconds += candidate_verification
            exact_evaluation_seconds += exact_evaluation

    best_raw = (
        min(candidates.values(), key=lambda candidate: candidate.objective)
        if candidates
        else None
    )

    repair_seconds = 0.0
    if settings.enable_repair:
        repair_indices = [
            index for index, report in enumerate(reports) if not report.feasible
        ]
        if settings.repair_candidate_limit is not None:
            repair_indices = sorted(
                repair_indices,
                key=lambda index: (
                    violation_score(instance, raw[index], reports[index]),
                    index,
                ),
            )[: settings.repair_candidate_limit]
        for index in repair_indices:
            report = reports[index]
            repair_attempts += 1
            repair_start = perf_counter()
            repaired = repair_placement(
                instance,
                raw[index],
                model_probability=None if probabilities is None else probabilities[index],
                config=RepairConfig(max_moves=settings.repair_max_moves),
            )
            repair_seconds += perf_counter() - repair_start
            total_repair_moves += len(repaired.moves)
            if repaired.success:
                repair_successes += 1
                candidate_verification, exact_evaluation = _add_verified_candidate(
                    instance,
                    candidates,
                    repaired.placement,
                    source="repair",
                    proposal_index=index,
                    repair_moves=len(repaired.moves),
                )
                candidate_verification_seconds += candidate_verification
                exact_evaluation_seconds += exact_evaluation

    candidate_available_before_recovery = bool(candidates)
    recovery_seconds = 0.0
    if settings.enable_recovery and not candidates:
        recovery_invoked = True
        recovery_indices = [
            index for index, report in enumerate(reports) if not report.feasible
        ]
        recovery_indices = sorted(
            recovery_indices,
            key=lambda index: (
                violation_score(instance, raw[index], reports[index]),
                index,
            ),
        )
        if settings.recovery_candidate_limit is not None:
            recovery_indices = recovery_indices[: settings.recovery_candidate_limit]
        for index in recovery_indices:
            recovery_attempts += 1
            recovery_start = perf_counter()
            recovered = recover_from_proposal(
                instance,
                raw[index],
                model_probability=(
                    None if probabilities is None else probabilities[index]
                ),
                config=ProposalRecoveryConfig(
                    max_released_services=settings.recovery_max_released_services,
                ),
            )
            recovery_seconds += perf_counter() - recovery_start
            recovery_released_services += len(recovered.released_services)
            recovery_completion_steps += len(recovered.completion_order)
            if recovered.success and recovered.placement is not None:
                recovery_successes += 1
                candidate_verification, exact_evaluation = _add_verified_candidate(
                    instance,
                    candidates,
                    recovered.placement,
                    source="recovery",
                    proposal_index=index,
                    repair_moves=0,
                )
                candidate_verification_seconds += candidate_verification
                exact_evaluation_seconds += exact_evaluation
            else:
                reason = recovered.failure_reason or "unknown"
                recovery_failure_reasons[reason] = (
                    recovery_failure_reasons.get(reason, 0) + 1
                )

    best_pre_fallback = (
        min(candidates.values(), key=lambda candidate: candidate.objective)
        if candidates
        else None
    )

    fallback_invoked = False
    fallback_success = False
    fallback_nodes = 0
    fallback_seconds = 0.0
    if settings.enable_fallback and (settings.always_include_fallback or not candidates):
        fallback_invoked = True
        fallback_start = perf_counter()
        fallback = construct_feasible_placement(
            instance,
            config=ConstructiveFallbackConfig(
                max_search_nodes=settings.fallback_max_search_nodes
            ),
        )
        fallback_seconds = perf_counter() - fallback_start
        fallback_success = fallback.success
        fallback_nodes = fallback.search_nodes
        if fallback.placement is not None:
            candidate_verification, exact_evaluation = _add_verified_candidate(
                instance,
                candidates,
                fallback.placement,
                source="fallback",
                proposal_index=None,
                repair_moves=0,
            )
            candidate_verification_seconds += candidate_verification
            exact_evaluation_seconds += exact_evaluation

    selection_start = perf_counter()
    ordered = tuple(
        sorted(
            candidates.values(),
            key=lambda candidate: (
                candidate.objective,
                candidate.source,
                tuple(candidate.placement.tolist()),
            ),
        )
    )
    best = ordered[0] if ordered else None
    selection_seconds = perf_counter() - selection_start
    post_sampling_seconds = perf_counter() - post_sampling_start
    count = raw.shape[0]
    metrics = {
        "proposal_method": proposal_method,
        "num_raw_proposals": count,
        "raw_feasible_count": raw_feasible,
        "raw_feasible_rate": raw_feasible / count,
        "raw_any_feasible": raw_feasible > 0,
        **proposal_diagnostics,
        "best_raw_objective": None if best_raw is None else best_raw.objective,
        "raw_capacity_violation_count": raw_capacity_violations,
        "raw_capacity_violation_rate": raw_capacity_violations / count,
        "raw_link_violation_count": raw_link_violations,
        "raw_link_violation_rate": raw_link_violations / count,
        "repair_attempts": repair_attempts,
        "repair_successes": repair_successes,
        "repair_success_rate": (
            repair_successes / repair_attempts if repair_attempts else 0.0
        ),
        "total_repair_moves": total_repair_moves,
        "candidate_available_before_recovery": candidate_available_before_recovery,
        "recovery_invoked": recovery_invoked,
        "recovery_attempts": recovery_attempts,
        "recovery_successes": recovery_successes,
        "recovery_success_rate": (
            recovery_successes / recovery_attempts if recovery_attempts else 0.0
        ),
        "recovery_released_services": recovery_released_services,
        "recovery_completion_steps": recovery_completion_steps,
        "recovery_failure_reasons": recovery_failure_reasons,
        "pre_fallback_success": best_pre_fallback is not None,
        "best_pre_fallback_objective": (
            None if best_pre_fallback is None else best_pre_fallback.objective
        ),
        "best_pre_fallback_source": (
            None if best_pre_fallback is None else best_pre_fallback.source
        ),
        "fallback_invoked": fallback_invoked,
        "fallback_success": fallback_success,
        "fallback_search_nodes": fallback_nodes,
        "final_success": best is not None,
        "sampling_seconds": float(sampling_seconds),
        "proposal_preparation_seconds": float(proposal_preparation_seconds),
        "verification_seconds": verification_seconds + candidate_verification_seconds,
        "repair_seconds": repair_seconds,
        "recovery_seconds": recovery_seconds,
        "fallback_seconds": fallback_seconds,
        "exact_evaluation_seconds": exact_evaluation_seconds,
        "selection_seconds": selection_seconds,
        "post_sampling_seconds": post_sampling_seconds,
        "total_seconds": (
            float(sampling_seconds)
            + float(proposal_preparation_seconds)
            + post_sampling_seconds
        ),
        "inference_config": asdict(settings),
    }
    return SolveResult(
        instance_id=instance.instance_id,
        placement=None if best is None else best.placement.copy(),
        objective=None if best is None else best.objective,
        source="failure" if best is None else best.source,
        verified_candidates=ordered,
        metrics=metrics,
    )


def solve_with_model(
    model: nn.Module,
    instance: DeploymentInstance,
    schedule: CategoricalSchedule,
    feature_schema: GraphFeatureSchema,
    *,
    config: InferenceConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> SolveResult:
    """Run learned reverse sampling followed by hard feasibility processing."""

    settings = config or InferenceConfig()
    proposals, probabilities, sampling_seconds = sample_reverse_proposals(
        model,
        instance,
        schedule,
        feature_schema,
        config=settings,
        device=device,
        generator=generator,
    )
    return solve_from_proposals(
        instance,
        proposals,
        model_probabilities=probabilities,
        config=settings,
        sampling_seconds=sampling_seconds,
        proposal_method="learned_reverse",
    )


def solve_with_direct_predictor(
    model: nn.Module,
    instance: DeploymentInstance,
    feature_schema: GraphFeatureSchema,
    *,
    config: InferenceConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> SolveResult:
    """Run one-pass learned sampling followed by the shared hard pipeline."""

    settings = config or InferenceConfig()
    proposals, probabilities, sampling_seconds = sample_direct_proposals(
        model,
        instance,
        feature_schema,
        config=settings,
        device=device,
        generator=generator,
    )
    return solve_from_proposals(
        instance,
        proposals,
        model_probabilities=probabilities,
        config=settings,
        sampling_seconds=sampling_seconds,
        proposal_method="direct_categorical",
    )


def solve_fallback_only(
    instance: DeploymentInstance,
    *,
    max_search_nodes: int = 100_000,
) -> SolveResult:
    """Run the deterministic constructive baseline without learned proposals."""

    start = perf_counter()
    fallback_start = perf_counter()
    fallback = construct_feasible_placement(
        instance,
        config=ConstructiveFallbackConfig(max_search_nodes=max_search_nodes),
    )
    fallback_seconds = perf_counter() - fallback_start
    candidates: dict[tuple[int, ...], VerifiedCandidate] = {}
    verification_seconds = 0.0
    exact_evaluation_seconds = 0.0
    if fallback.placement is not None:
        verification_seconds, exact_evaluation_seconds = _add_verified_candidate(
            instance,
            candidates,
            fallback.placement,
            source="fallback",
            proposal_index=None,
            repair_moves=0,
        )
    selection_start = perf_counter()
    ordered = tuple(candidates.values())
    best = ordered[0] if ordered else None
    selection_seconds = perf_counter() - selection_start
    elapsed = perf_counter() - start
    return SolveResult(
        instance_id=instance.instance_id,
        placement=None if best is None else best.placement.copy(),
        objective=None if best is None else best.objective,
        source="failure" if best is None else "fallback",
        verified_candidates=ordered,
        metrics={
            "proposal_method": "fallback_only",
            "num_raw_proposals": 0,
            "raw_feasible_count": 0,
            "raw_feasible_rate": None,
            "raw_capacity_violation_count": 0,
            "raw_capacity_violation_rate": None,
            "raw_link_violation_count": 0,
            "raw_link_violation_rate": None,
            "repair_attempts": 0,
            "repair_successes": 0,
            "repair_success_rate": 0.0,
            "total_repair_moves": 0,
            "fallback_invoked": True,
            "fallback_success": fallback.success,
            "fallback_search_nodes": fallback.search_nodes,
            "final_success": best is not None,
            "sampling_seconds": 0.0,
            "verification_seconds": verification_seconds,
            "repair_seconds": 0.0,
            "fallback_seconds": fallback_seconds,
            "exact_evaluation_seconds": exact_evaluation_seconds,
            "selection_seconds": selection_seconds,
            "post_sampling_seconds": elapsed,
            "total_seconds": elapsed,
        },
    )
