"""Checkpoint-selection evidence for Stage 3 masked and direct models."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Any

import torch

from gdm_factor_diffusion.common.seed import derive_seed
from gdm_factor_diffusion.diffusion import AbsorbingMaskSchedule
from gdm_factor_diffusion.graph import GraphFeatureSchema
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    solve_with_direct_predictor,
    solve_with_masked_model,
)

from .data import LabeledDeploymentDataset


@dataclass(frozen=True, slots=True)
class Stage3SelectionConfig:
    instance_limit: int = 64
    sample_batch_size: int = 8
    temperature: float = 1.0
    repair_max_moves: int = 10
    fallback_max_search_nodes: int = 100_000

    def validate(self) -> None:
        if self.instance_limit < 1 or self.sample_batch_size < 1:
            raise ValueError("Selection instance and batch limits must be positive.")
        if self.temperature <= 0 or self.repair_max_moves < 0:
            raise ValueError("Selection temperature or repair limit is invalid.")
        if self.fallback_max_search_nodes < 1:
            raise ValueError("fallback_max_search_nodes must be positive.")


@torch.no_grad()
def evaluate_stage3_selection(
    model: torch.nn.Module,
    feature_schema: GraphFeatureSchema,
    dataset: LabeledDeploymentDataset,
    *,
    model_kind: str,
    config: Stage3SelectionConfig,
    seed: int,
    device: torch.device | str,
    mask_schedule: AbsorbingMaskSchedule | None = None,
) -> dict[str, Any]:
    """Evaluate checkpoint candidates without touching the pilot-gate split."""

    config.validate()
    if model_kind not in {"masked_conditional", "direct"}:
        raise ValueError("Unsupported Stage 3 model kind.")
    if model_kind == "masked_conditional" and mask_schedule is None:
        raise ValueError("Masked validation requires an absorbing-MASK schedule.")
    count = min(len(dataset), config.instance_limit)
    records: list[dict[str, Any]] = []
    for index in range(count):
        item = dataset[index]
        generator = torch.Generator(device=device).manual_seed(
            derive_seed(seed, f"stage3-selection:{item.instance.instance_id}")
        )
        inference = InferenceConfig(
            num_samples=1,
            sample_batch_size=1,
            repair_max_moves=config.repair_max_moves,
            fallback_max_search_nodes=config.fallback_max_search_nodes,
            enable_repair=True,
            enable_fallback=True,
            always_include_fallback=False,
        )
        start = perf_counter()
        if model_kind == "masked_conditional":
            assert mask_schedule is not None
            result = solve_with_masked_model(
                model,
                item.instance,
                mask_schedule,
                feature_schema,
                decode_config=MaskedDecodeConfig(
                    num_samples=1,
                    sample_batch_size=1,
                    stochastic=False,
                    temperature=config.temperature,
                ),
                inference_config=inference,
                device=device,
                generator=generator,
            )
        else:
            result = solve_with_direct_predictor(
                model,
                item.instance,
                feature_schema,
                config=inference,
                device=device,
                generator=generator,
            )
        elapsed = perf_counter() - start
        pool_best = float(item.pool.latencies[0])
        pre_fallback = result.metrics["best_pre_fallback_objective"]
        records.append(
            {
                "instance_id": item.instance.instance_id,
                "final_success": result.success,
                "pre_fallback_success": pre_fallback is not None,
                "pre_fallback_gap": (
                    None if pre_fallback is None else pre_fallback / pool_best - 1.0
                ),
                "raw_any_feasible": bool(result.metrics["raw_any_feasible"]),
                "online_seconds": elapsed,
            }
        )
    pre_fallback_records = [
        record for record in records if record["pre_fallback_success"]
    ]
    return {
        "model_kind": model_kind,
        "instances": count,
        "final_verified_rate": mean(record["final_success"] for record in records),
        "pre_fallback_success_rate": len(pre_fallback_records) / count,
        "mean_pre_fallback_gap": (
            mean(record["pre_fallback_gap"] for record in pre_fallback_records)
            if pre_fallback_records
            else None
        ),
        "raw_any_feasibility": mean(
            record["raw_any_feasible"] for record in records
        ),
        "mean_online_seconds": mean(record["online_seconds"] for record in records),
        "records": records,
    }


def stage3_selection_rank(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    """Lexicographic rank that prevents fallback from hiding proposal failures."""

    gap = metrics["mean_pre_fallback_gap"]
    return (
        -float(metrics["pre_fallback_success_rate"]),
        float("inf") if gap is None else float(gap),
        -float(metrics["raw_any_feasibility"]),
        float(metrics["mean_online_seconds"]),
    )
