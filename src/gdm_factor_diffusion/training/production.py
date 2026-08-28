"""Resumable training streams and constrained validation for production runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

import torch

from gdm_factor_diffusion.common.seed import derive_seed
from gdm_factor_diffusion.diffusion import CategoricalSchedule
from gdm_factor_diffusion.graph import GraphFeatureSchema
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    solve_fallback_only,
    solve_with_direct_predictor,
    solve_with_model,
)

from .data import LabeledBatch, LabeledDeploymentDataset, LabeledItem


@dataclass(frozen=True, slots=True)
class ConstrainedValidationConfig:
    """Fixed bounded inference budget used for checkpoint selection."""

    num_samples: int = 4
    sample_batch_size: int = 4
    reverse_steps: int | None = 25
    repair_max_moves: int = 10
    fallback_max_search_nodes: int = 100_000
    instance_limit: int | None = None

    def validate(self) -> None:
        if self.num_samples < 1 or self.sample_batch_size < 1:
            raise ValueError("Validation sample counts must be positive.")
        if (
            self.reverse_steps is not None
            and self.reverse_steps < 1
        ) or self.repair_max_moves < 0:
            raise ValueError("Validation reverse steps and repair limit are invalid.")
        if self.fallback_max_search_nodes < 1:
            raise ValueError("fallback_max_search_nodes must be positive.")
        if self.instance_limit is not None and self.instance_limit < 1:
            raise ValueError("instance_limit must be positive when provided.")

    def inference_config(self) -> InferenceConfig:
        self.validate()
        return InferenceConfig(
            num_samples=self.num_samples,
            sample_batch_size=self.sample_batch_size,
            reverse_steps=self.reverse_steps,
            repair_max_moves=self.repair_max_moves,
            fallback_max_search_nodes=self.fallback_max_search_nodes,
            enable_repair=True,
            enable_fallback=True,
            always_include_fallback=True,
        )


def sample_training_batch(
    dataset: Sequence[LabeledItem],
    collate: Callable[[Sequence[LabeledItem]], LabeledBatch],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> LabeledBatch:
    """Sample one deterministic step batch without hidden DataLoader state."""

    if batch_size < 1 or batch_size > len(dataset):
        raise ValueError("batch_size must lie between one and the dataset size.")
    indices = torch.randperm(len(dataset), generator=generator)[:batch_size].tolist()
    return collate([dataset[index] for index in indices])


def capture_random_state(
    generators: Mapping[str, torch.Generator],
) -> dict[str, Any]:
    """Capture every random stream used by the step-based training contract."""

    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "generators": {
            name: generator.get_state() for name, generator in generators.items()
        },
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(
    state: Mapping[str, Any],
    generators: Mapping[str, torch.Generator],
) -> None:
    """Restore global and named random streams from a trusted checkpoint."""

    saved_generators = state.get("generators", {})
    if set(saved_generators) != set(generators):
        raise ValueError("Checkpoint and runtime named random streams disagree.")
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [device_state.cpu() for device_state in state["torch_cuda"]]
        )
    for name, generator in generators.items():
        generator.set_state(saved_generators[name].cpu())


@torch.no_grad()
def evaluate_constrained_validation(
    model: torch.nn.Module,
    schedule: CategoricalSchedule,
    feature_schema: GraphFeatureSchema,
    dataset: LabeledDeploymentDataset,
    *,
    config: ConstrainedValidationConfig | None = None,
    device: torch.device | str = "cpu",
    seed: int = 0,
    model_kind: str = "diffusion",
) -> dict[str, Any]:
    """Evaluate verified exact-latency performance on a fixed labeled subset."""

    settings = config or ConstrainedValidationConfig()
    inference_config = settings.inference_config()
    count = len(dataset)
    if settings.instance_limit is not None:
        count = min(count, settings.instance_limit)
    records: list[dict[str, Any]] = []
    learned_wins = 0
    learned_ties = 0
    for index in range(count):
        item = dataset[index]
        generator = torch.Generator(device=device).manual_seed(
            derive_seed(seed, f"constrained-validation:{item.instance.instance_id}")
        )
        if model_kind == "diffusion":
            learned = solve_with_model(
                model,
                item.instance,
                schedule,
                feature_schema,
                config=inference_config,
                device=device,
                generator=generator,
            )
        elif model_kind == "direct":
            learned = solve_with_direct_predictor(
                model,
                item.instance,
                feature_schema,
                config=inference_config,
                device=device,
                generator=generator,
            )
        else:
            raise ValueError("model_kind must be 'diffusion' or 'direct'.")
        fallback = solve_fallback_only(
            item.instance,
            max_search_nodes=settings.fallback_max_search_nodes,
        )
        pool_best = float(item.pool.latencies[0])
        objective = learned.objective
        fallback_objective = fallback.objective
        if objective is not None and fallback_objective is not None:
            if objective < fallback_objective - 1e-12:
                learned_wins += 1
            elif abs(objective - fallback_objective) <= 1e-12:
                learned_ties += 1
        records.append(
            {
                "instance_id": item.instance.instance_id,
                "success": learned.success,
                "objective": objective,
                "gap_to_pool_best": (
                    None if objective is None else objective / pool_best - 1.0
                ),
                "raw_feasible_rate": learned.metrics["raw_feasible_rate"],
                "fallback_objective": fallback_objective,
            }
        )
    successes = [record for record in records if record["success"]]
    return {
        "config": asdict(settings),
        "instances": count,
        "verified_rate": len(successes) / count,
        "mean_gap_to_pool_best": (
            mean(record["gap_to_pool_best"] for record in successes)
            if successes
            else None
        ),
        "mean_objective": (
            mean(record["objective"] for record in successes) if successes else None
        ),
        "mean_raw_feasible_rate": mean(
            record["raw_feasible_rate"] for record in records
        ),
        "learned_wins_over_fallback": learned_wins,
        "learned_ties_with_fallback": learned_ties,
        "records": records,
    }


def validation_rank(
    constrained_metrics: Mapping[str, Any],
    denoising_loss: float,
) -> tuple[float, float, float]:
    """Return the lexicographic checkpoint rank; smaller is better."""

    verified_rate = float(constrained_metrics["verified_rate"])
    gap = constrained_metrics.get("mean_gap_to_pool_best")
    return (
        -verified_rate,
        float("inf") if gap is None else float(gap),
        float(denoising_loss),
    )
