"""Run a small no-retraining Go/No-Go diagnostic for Phase 4C."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import create_run_directory, write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.diffusion import CategoricalSchedule
from gdm_factor_diffusion.graph import GraphFeatureSchema, build_factor_graph_batch
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    sample_reverse_proposals,
    solve_fallback_only,
    solve_from_proposals,
)
from gdm_factor_diffusion.models import DenoiserConfig, TypedFactorDenoiser
from gdm_factor_diffusion.training import LabeledDeploymentDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _load_model(
    checkpoint_path: Path,
    dataset: LabeledDeploymentDataset,
    device: torch.device,
) -> tuple[TypedFactorDenoiser, CategoricalSchedule, GraphFeatureSchema]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    metadata = checkpoint["metadata"]
    schema_payload = metadata["feature_schema"]
    feature_schema = GraphFeatureSchema(
        service_feature_names=tuple(schema_payload["service_feature_names"]),
        device_feature_names=tuple(schema_payload["device_feature_names"]),
        resource_names=tuple(schema_payload["resource_names"]),
    )
    reference = build_factor_graph_batch(
        [dataset[0].instance], feature_schema=feature_schema
    ).to(device)
    saved_config = metadata["config"]
    model_config = saved_config["model"]
    model = TypedFactorDenoiser.from_batch(
        reference,
        DenoiserConfig(
            num_diffusion_steps=int(saved_config["diffusion"]["steps"]),
            hidden_dim=int(model_config["hidden_dim"]),
            num_layers=int(model_config["layers"]),
            dropout=float(model_config["dropout"]),
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    schedule = CategoricalSchedule.from_betas(checkpoint["schedule_betas"].to(device))
    return model, schedule, feature_schema


def _record(
    *,
    instance_id: str,
    partition: str,
    inference_seed: int | None,
    reverse_steps: int | None,
    method: str,
    pool_best: float,
    result,
) -> dict:
    return {
        "instance_id": instance_id,
        "partition": partition,
        "inference_seed": inference_seed,
        "reverse_steps": reverse_steps,
        "method": method,
        "success": result.success,
        "source": result.source,
        "objective": result.objective,
        "pool_best": pool_best,
        "gap_to_pool_best": (
            None if result.objective is None else result.objective / pool_best - 1.0
        ),
        "metrics": result.metrics,
    }


def _aggregate(records: list[dict]) -> dict:
    successes = [record for record in records if record["success"]]
    sources = Counter(record["source"] for record in records)
    raw_values = [
        record["metrics"]["raw_feasible_rate"]
        for record in records
        if record["metrics"]["raw_feasible_rate"] is not None
    ]
    return {
        "instances": len(records),
        "final_success_rate": len(successes) / len(records),
        "mean_gap_to_pool_best": (
            mean(record["gap_to_pool_best"] for record in successes)
            if successes
            else None
        ),
        "mean_raw_feasible_rate": mean(raw_values) if raw_values else None,
        "mean_total_seconds": mean(
            record["metrics"]["total_seconds"] for record in records
        ),
        "selection_sources": dict(sorted(sources.items())),
    }


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (
        implementation_root / "configs" / "phase4c_small_validation.yaml"
    )
    config = load_config(config_path)
    evaluation = config["evaluation"]
    diagnostic = config["diagnostic"]
    seed = int(evaluation["seed"])
    seed_everything(seed)
    device = torch.device(
        evaluation["device"] if torch.cuda.is_available() else "cpu"
    )
    dataset_root = _resolve(implementation_root, evaluation["dataset_root"])
    checkpoint_path = _resolve(implementation_root, evaluation["checkpoint"])
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=tuple(evaluation["partitions"]),
    )
    model, schedule, feature_schema = _load_model(checkpoint_path, dataset, device)
    requested_steps = tuple(int(value) for value in diagnostic["reverse_steps"])
    inference_seeds = tuple(int(value) for value in diagnostic["seeds"])
    if any(value > schedule.num_steps for value in requested_steps):
        raise ValueError("A requested reverse-step budget exceeds the trained schedule.")

    run_root = args.output or implementation_root / "artifacts" / "runs"
    run_directory = create_run_directory(run_root, "phase4c-small")
    write_json(run_directory / "config.json", config)
    records: list[dict] = []
    for item in (dataset[index] for index in range(len(dataset))):
        instance = item.instance
        pool_best = float(item.pool.latencies[0])
        fallback = solve_fallback_only(
            instance,
            max_search_nodes=int(diagnostic["fallback_max_search_nodes"]),
        )
        records.append(
            _record(
                instance_id=instance.instance_id,
                partition=item.partition,
                inference_seed=None,
                reverse_steps=None,
                method="fallback_only",
                pool_best=pool_best,
                result=fallback,
            )
        )
        for inference_seed in inference_seeds:
            for reverse_steps in requested_steps:
                common = dict(
                    num_samples=int(diagnostic["num_samples"]),
                    sample_batch_size=int(diagnostic["sample_batch_size"]),
                    repair_max_moves=int(diagnostic["repair_max_moves"]),
                    fallback_max_search_nodes=int(
                        diagnostic["fallback_max_search_nodes"]
                    ),
                    reverse_steps=reverse_steps,
                )
                proposal_config = InferenceConfig(
                    **common, enable_repair=True, enable_fallback=False
                )
                generator = torch.Generator(device=device).manual_seed(
                    derive_seed(
                        inference_seed,
                        f"{instance.instance_id}:steps={reverse_steps}",
                    )
                )
                proposals, probabilities, sampling_seconds = sample_reverse_proposals(
                    model,
                    instance,
                    schedule,
                    feature_schema,
                    config=proposal_config,
                    device=device,
                    generator=generator,
                )
                variants = (
                    (
                        "learned_raw_only",
                        InferenceConfig(
                            **common, enable_repair=False, enable_fallback=False
                        ),
                    ),
                    (
                        "learned_repair",
                        InferenceConfig(
                            **common, enable_repair=True, enable_fallback=False
                        ),
                    ),
                    (
                        "learned_hybrid",
                        InferenceConfig(
                            **common,
                            enable_repair=True,
                            enable_fallback=True,
                            always_include_fallback=True,
                        ),
                    ),
                )
                for method, settings in variants:
                    result = solve_from_proposals(
                        instance,
                        proposals,
                        model_probabilities=probabilities,
                        config=settings,
                        sampling_seconds=sampling_seconds,
                        proposal_method=f"{method}:{reverse_steps}",
                    )
                    records.append(
                        _record(
                            instance_id=instance.instance_id,
                            partition=item.partition,
                            inference_seed=inference_seed,
                            reverse_steps=reverse_steps,
                            method=method,
                            pool_best=pool_best,
                            result=result,
                        )
                    )
                print(
                    f"instance={instance.instance_id} seed={inference_seed} "
                    f"steps={reverse_steps} sampling={sampling_seconds:.3f}s"
                )

    grouped: dict[str, dict] = {}
    for method in sorted({record["method"] for record in records}):
        method_records = [record for record in records if record["method"] == method]
        if method == "fallback_only":
            grouped[method] = _aggregate(method_records)
            continue
        grouped[method] = {
            str(reverse_steps): _aggregate(
                [
                    record
                    for record in method_records
                    if record["reverse_steps"] == reverse_steps
                ]
            )
            for reverse_steps in requested_steps
        }
    summary = {
        "run_directory": str(run_directory),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "scope": "small_no_retraining_diagnostic",
        "methods": grouped,
    }
    with (run_directory / "records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(run_directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
