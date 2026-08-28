"""Smoke-test learned reverse solving against random and fallback-only baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import create_run_directory, write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.diffusion import CategoricalSchedule
from gdm_factor_diffusion.graph import GraphFeatureSchema, build_factor_graph_batch
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    sample_random_proposals,
    solve_fallback_only,
    solve_from_proposals,
    solve_with_model,
)
from gdm_factor_diffusion.models import DenoiserConfig, TypedFactorDenoiser
from gdm_factor_diffusion.training import LabeledDeploymentDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _aggregate(records: list[dict], method: str) -> dict:
    selected = [record for record in records if record["method"] == method]
    successes = [record for record in selected if record["success"]]
    def optional_mean(metric: str) -> float | None:
        values = [
            record["metrics"][metric]
            for record in selected
            if record["metrics"][metric] is not None
        ]
        return mean(values) if values else None

    return {
        "instances": len(selected),
        "final_success_rate": len(successes) / len(selected),
        "mean_objective": (
            mean(record["objective"] for record in successes) if successes else None
        ),
        "mean_gap_to_pool_best": (
            mean(record["gap_to_pool_best"] for record in successes)
            if successes
            else None
        ),
        "mean_raw_feasible_rate": optional_mean("raw_feasible_rate"),
        "mean_raw_capacity_violation_rate": optional_mean(
            "raw_capacity_violation_rate"
        ),
        "mean_raw_link_violation_rate": optional_mean("raw_link_violation_rate"),
        "mean_total_seconds": mean(
            record["metrics"]["total_seconds"] for record in selected
        ),
        "fallback_invocation_rate": mean(
            float(record["metrics"]["fallback_invoked"]) for record in selected
        ),
    }


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (
        implementation_root / "configs" / "inference_phase4b_smoke.yaml"
    )
    config = load_config(config_path)
    evaluation = config["evaluation"]
    inference = config["inference"]
    seed = int(evaluation["seed"])
    seed_everything(seed)
    device = torch.device(
        evaluation["device"] if torch.cuda.is_available() else "cpu"
    )
    dataset_root = _resolve(implementation_root, evaluation["dataset_root"])
    checkpoint_path = args.checkpoint or _resolve(
        implementation_root, evaluation["checkpoint"]
    )
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=tuple(evaluation["partitions"]),
    )

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
    schedule = CategoricalSchedule.from_betas(
        checkpoint["schedule_betas"].to(device)
    )
    inference_config = InferenceConfig(
        num_samples=int(inference["num_samples"]),
        sample_batch_size=int(inference["sample_batch_size"]),
        repair_max_moves=int(inference["repair_max_moves"]),
        fallback_max_search_nodes=int(inference["fallback_max_search_nodes"]),
        enable_repair=bool(inference["enable_repair"]),
        enable_fallback=bool(inference["enable_fallback"]),
        always_include_fallback=bool(
            inference.get("always_include_fallback", False)
        ),
        reverse_steps=(
            None
            if inference.get("reverse_steps") is None
            else int(inference["reverse_steps"])
        ),
    )
    inference_config.validate()

    run_root = args.output or implementation_root / "artifacts" / "runs"
    run_directory = create_run_directory(run_root, "phase4b-inference")
    write_json(run_directory / "config.json", config)
    records: list[dict] = []
    learned_method = (
        "learned_hybrid"
        if inference_config.always_include_fallback
        else "learned"
    )
    random_method = (
        "random_hybrid"
        if inference_config.always_include_fallback
        else "random"
    )
    for index in range(len(dataset)):
        item = dataset[index]
        instance = item.instance
        learned_generator = torch.Generator(device=device).manual_seed(
            derive_seed(seed, f"learned:{instance.instance_id}")
        )
        random_generator = torch.Generator().manual_seed(
            derive_seed(seed, f"random:{instance.instance_id}")
        )
        learned = solve_with_model(
            model,
            instance,
            schedule,
            feature_schema,
            config=inference_config,
            device=device,
            generator=learned_generator,
        )
        random_start = perf_counter()
        random_proposals = sample_random_proposals(
            instance,
            num_samples=inference_config.num_samples,
            generator=random_generator,
        )
        random_sampling_seconds = perf_counter() - random_start
        random_result = solve_from_proposals(
            instance,
            random_proposals,
            config=inference_config,
            sampling_seconds=random_sampling_seconds,
            proposal_method="random_categorical",
        )
        fallback = solve_fallback_only(
            instance,
            max_search_nodes=inference_config.fallback_max_search_nodes,
        )
        pool_best = float(item.pool.latencies[0])
        for method, result in (
            (learned_method, learned),
            (random_method, random_result),
            ("fallback_only", fallback),
        ):
            objective = result.objective
            record = {
                "instance_id": instance.instance_id,
                "partition": item.partition,
                "method": method,
                "pool_best": pool_best,
                "success": result.success,
                "source": result.source,
                "objective": objective,
                "gap_to_pool_best": (
                    None if objective is None else objective / pool_best - 1.0
                ),
                "metrics": result.metrics,
            }
            records.append(record)
        print(
            f"instance={instance.instance_id} "
            f"learned={learned.objective} random={random_result.objective} "
            f"fallback={fallback.objective} "
            f"learned_raw_feasible={learned.metrics['raw_feasible_rate']:.3f}"
        )

    methods = (learned_method, random_method, "fallback_only")
    summary = {
        "run_directory": str(run_directory),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "methods": {method: _aggregate(records, method) for method in methods},
        "partitions": {
            partition: {
                method: _aggregate(
                    [record for record in records if record["partition"] == partition],
                    method,
                )
                for method in methods
            }
            for partition in sorted({record["partition"] for record in records})
        },
    }
    with (run_directory / "records.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(run_directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
