"""Read-only diagnostics for noisy-state use and reverse-trajectory quality."""

from __future__ import annotations

from dataclasses import dataclass
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
)
from gdm_factor_diffusion.graph import GraphFeatureSchema, build_factor_graph_batch
from gdm_factor_diffusion.solver import evaluate_latency, verify_placement


@dataclass(frozen=True, slots=True)
class TrajectoryDiagnosticConfig:
    num_samples: int = 4
    reverse_steps: int = 25
    anchor_count: int = 5

    def validate(self, schedule: CategoricalSchedule) -> None:
        if self.num_samples < 2:
            raise ValueError("Trajectory diagnostics require at least two samples.")
        if not 1 <= self.reverse_steps <= schedule.num_steps:
            raise ValueError("reverse_steps must lie in [1, schedule.num_steps].")
        if not 1 <= self.anchor_count <= self.reverse_steps:
            raise ValueError("anchor_count must lie in [1, reverse_steps].")


def _anchor_indices(num_transitions: int, anchor_count: int) -> tuple[int, ...]:
    values = np.linspace(0, num_transitions - 1, anchor_count)
    return tuple(sorted({int(round(value)) for value in values}))


def _js_divergence(probability: Tensor, alternative: Tensor) -> Tensor:
    mixture = 0.5 * (probability + alternative)
    tiny = torch.finfo(probability.dtype).tiny
    first = torch.where(
        probability > 0,
        probability * (probability.clamp_min(tiny).log() - mixture.clamp_min(tiny).log()),
        torch.zeros_like(probability),
    )
    second = torch.where(
        alternative > 0,
        alternative * (
            alternative.clamp_min(tiny).log() - mixture.clamp_min(tiny).log()
        ),
        torch.zeros_like(alternative),
    )
    return 0.5 * (first + second).sum(dim=-1)


def _distribution_change(
    baseline_logits: Tensor,
    alternative_logits: Tensor,
    candidate_mask: Tensor,
    service_mask: Tensor,
) -> tuple[Tensor, float, float, float]:
    baseline = masked_softmax(baseline_logits, candidate_mask, service_mask)
    alternative = masked_softmax(alternative_logits, candidate_mask, service_mask)
    js = _js_divergence(baseline, alternative)
    active_js = js[service_mask]
    argmax_change = (
        baseline.argmax(dim=-1)[service_mask]
        != alternative.argmax(dim=-1)[service_mask]
    ).float()
    total_variation = 0.5 * (baseline - alternative).abs().sum(dim=-1)
    return (
        js,
        float(active_js.mean().item()),
        float(argmax_change.mean().item()),
        float(total_variation[service_mask].mean().item()),
    )


def _perturb_states(
    state: Tensor,
    candidate_mask: Tensor,
    instance: DeploymentInstance,
    *,
    anchor_index: int,
) -> tuple[Tensor, tuple[int, ...]]:
    perturbed = state.clone()
    eligible = np.flatnonzero(instance.compatibility_mask.sum(axis=1) > 1)
    if not eligible.size:
        raise ValueError("State perturbation requires one multi-candidate service.")
    changed: list[int] = []
    for row in range(state.shape[0]):
        service = int(eligible[(row + anchor_index) % eligible.size])
        candidates = torch.nonzero(
            candidate_mask[row, service], as_tuple=False
        ).flatten()
        current = int(state[row, service].item())
        position = int(torch.where(candidates == current)[0].item())
        replacement = int(candidates[(position + 1) % candidates.numel()].item())
        perturbed[row, service] = replacement
        changed.append(service)
    return perturbed, tuple(changed)


def _response_groups(
    service_js: Tensor,
    changed_services: tuple[int, ...],
    instance: DeploymentInstance,
) -> dict[str, float | int]:
    dependency_neighbors = [set() for _ in range(instance.num_services)]
    for source, target in instance.dependency_index.T:
        dependency_neighbors[int(source)].add(int(target))
        dependency_neighbors[int(target)].add(int(source))

    sums = {
        "target": 0.0,
        "neighbor": 0.0,
        "non_neighbor": 0.0,
        "competitor": 0.0,
        "unrelated": 0.0,
    }
    counts = {key: 0 for key in sums}
    for row, changed in enumerate(changed_services):
        neighbors = dependency_neighbors[changed]
        shared = set(
            np.flatnonzero(
                (
                    instance.compatibility_mask
                    & instance.compatibility_mask[changed][None, :]
                ).any(axis=1)
            ).tolist()
        )
        competitors = shared - neighbors - {changed}
        non_neighbors = set(range(instance.num_services)) - neighbors - {changed}
        unrelated = non_neighbors - competitors
        groups = {
            "target": {changed},
            "neighbor": neighbors,
            "non_neighbor": non_neighbors,
            "competitor": competitors,
            "unrelated": unrelated,
        }
        for group, services in groups.items():
            for service in services:
                sums[group] += float(service_js[row, service].item())
                counts[group] += 1
    return {
        **{f"{key}_js_sum": value for key, value in sums.items()},
        **{f"{key}_count": value for key, value in counts.items()},
    }


def _candidate_metrics(
    instance: DeploymentInstance,
    placements: np.ndarray,
    reference_objective: float,
) -> dict[str, Any]:
    raw = np.asarray(placements, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != instance.num_services:
        raise ValueError("Diagnostic placements must have shape [K, M].")
    reports = [verify_placement(instance, placement) for placement in raw]
    objectives = [
        evaluate_latency(instance, report.placement).objective
        for report in reports
        if report.feasible
    ]
    best_objective = min(objectives) if objectives else None
    best_gap = (
        None
        if best_objective is None
        else float(best_objective / reference_objective - 1.0)
    )
    relative_capacity_excess = [
        float(
            (
                report.capacity_excess
                / np.maximum(instance.device_capacity.astype(np.float64), 1e-8)
            ).sum()
        )
        for report in reports
    ]
    return {
        "candidate_count": int(raw.shape[0]),
        "unique_count": int(np.unique(raw, axis=0).shape[0]),
        "feasible_count": int(sum(report.feasible for report in reports)),
        "any_feasible": bool(objectives),
        "capacity_invalid_count": int(sum(not report.capacity_valid for report in reports)),
        "link_invalid_count": int(sum(not report.direct_link_valid for report in reports)),
        "mean_relative_capacity_excess": float(np.mean(relative_capacity_excess)),
        "mean_disconnected_dependencies": float(
            np.mean([len(report.disconnected_dependencies) for report in reports])
        ),
        "best_objective": best_objective,
        "best_gap_to_pool_best": best_gap,
    }


@torch.no_grad()
def diagnose_reverse_trajectory(
    model: nn.Module,
    instance: DeploymentInstance,
    schedule: CategoricalSchedule,
    feature_schema: GraphFeatureSchema,
    *,
    reference_objective: float,
    config: TrajectoryDiagnosticConfig | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Measure state sensitivity and candidate quality without changing sampling."""

    settings = config or TrajectoryDiagnosticConfig()
    settings.validate(schedule)
    target_device = torch.device(device)
    model.eval()
    batch = build_factor_graph_batch(
        [instance] * settings.num_samples,
        feature_schema=feature_schema,
    ).to(target_device)
    reverse_grid = build_reverse_timestep_grid(
        schedule.num_steps,
        settings.reverse_steps,
    )
    anchors = set(_anchor_indices(settings.reverse_steps, settings.anchor_count))
    snapshots: list[dict[str, Any]] = []
    reservoir: list[np.ndarray] = []
    callback_index = 0
    start = perf_counter()

    def observe(
        timestep: int,
        previous_timestep: int,
        noisy_state: Tensor,
        clean_logits: Tensor,
        previous_state: Tensor,
    ) -> None:
        nonlocal callback_index
        index = callback_index
        callback_index += 1
        if index not in anchors:
            return
        timestep_tensor = torch.full(
            (settings.num_samples,),
            timestep,
            dtype=torch.long,
            device=target_device,
        )
        shuffled_state = noisy_state.roll(1, dims=0)
        shuffled_logits = model(batch, shuffled_state, timestep_tensor)
        _, shuffle_js, shuffle_argmax, shuffle_tv = _distribution_change(
            clean_logits,
            shuffled_logits,
            batch.candidate_mask,
            batch.service_mask,
        )
        perturbed_state, changed = _perturb_states(
            noisy_state,
            batch.candidate_mask,
            instance,
            anchor_index=index,
        )
        perturbed_logits = model(batch, perturbed_state, timestep_tensor)
        service_js, perturb_js, perturb_argmax, perturb_tv = _distribution_change(
            clean_logits,
            perturbed_logits,
            batch.candidate_mask,
            batch.service_mask,
        )
        response = _response_groups(service_js, changed, instance)
        clean_argmax = clean_logits.argmax(dim=-1)
        sampled = previous_state[:, : instance.num_services].cpu().numpy()
        predicted = clean_argmax[:, : instance.num_services].cpu().numpy()
        reservoir.extend((sampled, predicted))
        snapshots.append(
            {
                "transition_index": index,
                "timestep": timestep,
                "previous_timestep": previous_timestep,
                "state_use": {
                    "shuffle_js": shuffle_js,
                    "shuffle_argmax_change": shuffle_argmax,
                    "shuffle_total_variation": shuffle_tv,
                    "perturb_js": perturb_js,
                    "perturb_argmax_change": perturb_argmax,
                    "perturb_total_variation": perturb_tv,
                },
                "dependency_response": response,
                "sampled_state": _candidate_metrics(
                    instance, sampled, reference_objective
                ),
                "clean_argmax": _candidate_metrics(
                    instance, predicted, reference_objective
                ),
            }
        )

    final_state = reverse_sample_loop(
        lambda noisy, timestep: model(batch, noisy, timestep),
        batch.candidate_mask,
        schedule,
        batch.service_mask,
        timesteps=reverse_grid,
        generator=generator,
        step_callback=observe,
    )
    elapsed = perf_counter() - start
    final_placements = final_state[:, : instance.num_services].cpu().numpy()
    reservoir_placements = np.concatenate(reservoir, axis=0)
    final_metrics = _candidate_metrics(
        instance, final_placements, reference_objective
    )
    reservoir_metrics = _candidate_metrics(
        instance, reservoir_placements, reference_objective
    )
    final_best = final_metrics["best_objective"]
    reservoir_best = reservoir_metrics["best_objective"]
    reservoir_improves = bool(
        reservoir_best is not None
        and (final_best is None or reservoir_best < final_best - 1e-12)
    )
    return {
        "diagnostic_schema_version": "1.1",
        "instance_id": instance.instance_id,
        "num_services": instance.num_services,
        "num_devices": instance.num_devices,
        "num_dependencies": instance.num_dependencies,
        "num_samples": settings.num_samples,
        "reverse_steps": settings.reverse_steps,
        "anchor_count": len(snapshots),
        "diagnostic_seconds": elapsed,
        "snapshots": snapshots,
        "final": final_metrics,
        "reservoir": reservoir_metrics,
        "reservoir_improves_final": reservoir_improves,
    }
