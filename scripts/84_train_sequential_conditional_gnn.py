"""Train and smoke-evaluate the Sequential Conditional GNN baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.graph import merge_feature_schemas
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    SequentialDecodeConfig,
    solve_with_sequential_model,
)
from gdm_factor_diffusion.models import (
    SequentialPolicyConfig,
    TypedFactorSequentialPolicy,
)
from gdm_factor_diffusion.sequence import service_order_batch
from gdm_factor_diffusion.training import (
    LabeledDeploymentDataset,
    SequentialConditionalTrainer,
    SequentialTrainerConfig,
    audit_dataset_freeze,
    capture_random_state,
    load_sequential_checkpoint,
    make_labeled_collator,
    restore_random_state,
    restore_sequential_checkpoint,
    sample_clean_targets,
    sample_training_batch,
    save_sequential_checkpoint,
)


@dataclass(frozen=True, slots=True)
class SequentialSelectionConfig:
    instance_limit: int = 64
    forward_equivalent_budget: int = 64
    sample_batch_size: int = 8
    temperature: float = 1.0
    fallback_max_search_nodes: int = 30_000

    def validate(self) -> None:
        if self.instance_limit < 1 or self.forward_equivalent_budget < 1:
            raise ValueError("Selection limits must be positive.")
        if self.sample_batch_size < 1 or self.temperature <= 0:
            raise ValueError("Selection sampling settings are invalid.")
        if self.fallback_max_search_nodes < 1:
            raise ValueError("fallback_max_search_nodes must be positive.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Stage 3-style YAML config; defaults to seed 2026070114.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--selection-instances", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root for sequential runs.",
    )
    return parser.parse_args()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _selection_config(
    config: dict[str, Any],
    *,
    override_instances: int | None,
) -> SequentialSelectionConfig:
    value = config.get("checkpoint_selection", {})
    return SequentialSelectionConfig(
        instance_limit=(
            int(override_instances)
            if override_instances is not None
            else int(value.get("instance_limit", 64))
        ),
        forward_equivalent_budget=int(value.get("forward_equivalent_budget", 64)),
        sample_batch_size=int(value.get("sample_batch_size", 8)),
        temperature=float(value.get("temperature", 1.0)),
        fallback_max_search_nodes=int(value.get("fallback_max_search_nodes", 30_000)),
    )


def _proposal_count(instance_services: int, config: SequentialSelectionConfig) -> int:
    return max(1, config.forward_equivalent_budget // max(1, int(instance_services)))


@torch.no_grad()
def evaluate_sequential_selection(
    model: torch.nn.Module,
    feature_schema,
    dataset: LabeledDeploymentDataset,
    *,
    config: SequentialSelectionConfig,
    seed: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Evaluate checkpoints under the no-repair verify+fallback policy."""

    config.validate()
    target_device = torch.device(device)
    count = min(len(dataset), config.instance_limit)
    records: list[dict[str, Any]] = []
    for index in range(count):
        item = dataset[index]
        samples = _proposal_count(item.instance.num_services, config)
        generator = torch.Generator(device=target_device).manual_seed(
            derive_seed(seed, f"sequential-selection:{item.instance.instance_id}")
        )
        inference = InferenceConfig(
            num_samples=samples,
            sample_batch_size=min(config.sample_batch_size, samples),
            fallback_max_search_nodes=config.fallback_max_search_nodes,
            enable_repair=False,
            enable_fallback=True,
            always_include_fallback=False,
        )
        start = perf_counter()
        result = solve_with_sequential_model(
            model,
            item.instance,
            feature_schema,
            decode_config=SequentialDecodeConfig(
                num_samples=samples,
                sample_batch_size=min(config.sample_batch_size, samples),
                stochastic=True,
                temperature=config.temperature,
            ),
            inference_config=inference,
            device=target_device,
            generator=generator,
        )
        elapsed = perf_counter() - start
        pool_best = float(np.min(item.pool.latencies))
        raw_objective = result.metrics["best_raw_objective"]
        records.append(
            {
                "instance_id": item.instance.instance_id,
                "samples": samples,
                "neural_forward_budget": samples * item.instance.num_services,
                "final_success": result.success,
                "raw_success": raw_objective is not None,
                "raw_gap": (
                    None if raw_objective is None else float(raw_objective) / pool_best - 1.0
                ),
                "final_gap": (
                    None if result.objective is None else float(result.objective) / pool_best - 1.0
                ),
                "raw_feasible_rate": result.metrics["raw_feasible_rate"],
                "raw_any_feasible": bool(result.metrics["raw_any_feasible"]),
                "fallback_invoked": bool(result.metrics["fallback_invoked"]),
                "completed_rate": result.metrics["sequential_completed_rate"],
                "online_seconds": elapsed,
                "selected_source": result.source,
            }
        )
    raw_records = [record for record in records if record["raw_success"]]
    final_records = [record for record in records if record["final_success"]]
    return {
        "model_kind": "sequential_conditional",
        "instances": count,
        "selection_config": asdict(config),
        "final_verified_rate": mean(record["final_success"] for record in records),
        "raw_success_rate": len(raw_records) / count,
        "raw_any_feasibility": mean(record["raw_any_feasible"] for record in records),
        "proposal_feasible_rate": mean(record["raw_feasible_rate"] for record in records),
        "mean_raw_gap": (
            None if not raw_records else mean(record["raw_gap"] for record in raw_records)
        ),
        "mean_final_gap": (
            None if not final_records else mean(record["final_gap"] for record in final_records)
        ),
        "fallback_invocation_rate": mean(record["fallback_invoked"] for record in records),
        "mean_completed_rate": mean(record["completed_rate"] for record in records),
        "mean_online_seconds": mean(record["online_seconds"] for record in records),
        "mean_neural_forward_budget": mean(
            record["neural_forward_budget"] for record in records
        ),
        "records": records,
    }


def selection_rank(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    gap = metrics["mean_raw_gap"]
    return (
        -float(metrics["raw_success_rate"]),
        float("inf") if gap is None else float(gap),
        -float(metrics["proposal_feasible_rate"]),
        float(metrics["mean_online_seconds"]),
    )


def _model_config(config: dict[str, Any]) -> SequentialPolicyConfig:
    value = config.get("sequential_control") or config.get("direct_control") or config["model"]
    return SequentialPolicyConfig(
        hidden_dim=int(value.get("hidden_dim", 128)),
        num_layers=int(value.get("num_layers", 4)),
        dropout=float(value.get("dropout", 0.0)),
    )


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (
        implementation_root
        / "configs"
        / "training_phase6e_e_stage39_seed2026070114.yaml"
    )
    config = load_config(config_path)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    dataset_root = _resolve(implementation_root, config["experiment"]["dataset_root"])
    freeze = audit_dataset_freeze(dataset_root)
    train_dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=(config["experiment"]["train_partition"],),
        require_freeze=True,
    )
    checkpoint_dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=(config["experiment"]["checkpoint_partition"],),
        require_freeze=True,
    )
    feature_schema = merge_feature_schemas(
        (train_dataset.feature_schema, checkpoint_dataset.feature_schema)
    )
    collate = make_labeled_collator(feature_schema)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = collate([train_dataset[0]]).factor_graph.to(device)
    model_config = _model_config(config)
    model = TypedFactorSequentialPolicy.from_batch(reference, model_config).to(device)
    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    trainer = SequentialConditionalTrainer(
        model,
        optimizer,
        SequentialTrainerConfig(
            gradient_clip_norm=float(optimization["gradient_clip_norm"])
        ),
    )
    streams = {
        "batch": torch.Generator().manual_seed(derive_seed(seed, "sequential-batch")),
        "target": torch.Generator().manual_seed(derive_seed(seed, "sequential-target")),
        "step": torch.Generator(device=device).manual_seed(
            derive_seed(seed, "sequential-step")
        ),
    }
    target_mode = str(config["experiment"].get("target_mode", "best"))
    selection_config = _selection_config(
        config,
        override_instances=args.selection_instances,
    )

    if args.preflight:
        batch = collate([train_dataset[0], train_dataset[1]])
        target = sample_clean_targets(batch, mode=target_mode, generator=streams["target"])
        graph = batch.factor_graph.to(device)
        order = service_order_batch(
            [item.instance for item in batch.items],
            max_services=graph.candidate_mask.shape[1],
        ).to(device)
        terms = trainer.evaluate_step(
            graph,
            target.state.to(device),
            order,
            torch.zeros(graph.batch_size, dtype=torch.long, device=device),
        )
        smoke_config = SequentialSelectionConfig(
            instance_limit=1,
            forward_equivalent_budget=selection_config.forward_equivalent_budget,
            sample_batch_size=selection_config.sample_batch_size,
            temperature=selection_config.temperature,
            fallback_max_search_nodes=selection_config.fallback_max_search_nodes,
        )
        selection = evaluate_sequential_selection(
            model,
            feature_schema,
            checkpoint_dataset,
            config=smoke_config,
            seed=seed,
            device=device,
        )
        report = {
            "schema_version": "1.0",
            "mode": "preflight_no_optimizer_step",
            "model_kind": "sequential_conditional",
            "device": str(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "dataset_freeze": freeze["core_sha256"],
            "batch_loss": float(terms.total.item()),
            "selection_smoke": selection,
            "passed": torch.isfinite(terms.total).item() and selection["instances"] == 1,
        }
        destination = (
            implementation_root
            / "artifacts"
            / "phase6f-sequential-conditional-training"
            / f"preflight_seed{seed}.json"
        )
        write_json(destination, report)
        print(
            f"model_kind=sequential_conditional mode=preflight device={device} "
            f"loss={float(terms.total.item()):.6f} passed={report['passed']} "
            f"report={destination}"
        )
        return

    output_root = args.output_root or (
        implementation_root / "artifacts" / "phase6f-sequential-conditional-training"
    )
    run_directory = _resolve(output_root, f"sequential_conditional-seed{seed}")
    run_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = run_directory / "metrics.jsonl"
    if args.resume is None and metrics_path.exists():
        raise FileExistsError(
            f"Run already exists; pass --resume to continue: {run_directory}"
        )
    max_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else int(optimization["max_steps"])
    )
    validation_interval = (
        int(args.validation_interval)
        if args.validation_interval is not None
        else int(optimization["validation_interval"])
    )
    checkpoint_interval = (
        int(args.checkpoint_interval)
        if args.checkpoint_interval is not None
        else int(optimization["checkpoint_interval"])
    )
    best_rank: tuple[float, ...] | None = None

    def metadata(selection: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "config": config,
            "model_kind": "sequential_conditional",
            "model_config": asdict(model_config),
            "feature_schema": {
                "service_feature_names": list(feature_schema.service_feature_names),
                "device_feature_names": list(feature_schema.device_feature_names),
                "resource_names": list(feature_schema.resource_names),
            },
            "service_order": "topological_order",
            "selection_policy": "verified_raw_then_fallback_no_repair",
            "selection_config": asdict(selection_config),
            "best_rank": best_rank,
            "latest_selection": selection,
        }

    def save_named(name: str, selection: dict[str, Any] | None = None) -> Path:
        return save_sequential_checkpoint(
            run_directory / name,
            trainer,
            metadata=metadata(selection),
            runtime_state=capture_random_state(streams),
        )

    if args.resume is not None:
        payload = restore_sequential_checkpoint(args.resume, trainer, map_location=device)
        restore_random_state(payload["runtime_state"], streams)
        saved_rank = payload["metadata"].get("best_rank")
        best_rank = None if saved_rank is None else tuple(saved_rank)
    write_json(run_directory / "config.json", config)

    while trainer.step < max_steps:
        batch = sample_training_batch(
            train_dataset,
            collate,
            batch_size=int(optimization["batch_size"]),
            generator=streams["batch"],
        )
        target = sample_clean_targets(batch, mode=target_mode, generator=streams["target"])
        graph = batch.factor_graph.to(device)
        order = service_order_batch(
            [item.instance for item in batch.items],
            max_services=graph.candidate_mask.shape[1],
        ).to(device)
        metrics = trainer.train_step(
            graph,
            target.state.to(device),
            order,
            generator=streams["step"],
        )
        if trainer.step == 1 or trainer.step % 50 == 0:
            _append_jsonl(metrics_path, {"type": "train", **metrics})

        selection = None
        if trainer.step % validation_interval == 0:
            selection = evaluate_sequential_selection(
                model,
                feature_schema,
                checkpoint_dataset,
                config=selection_config,
                seed=derive_seed(seed, f"sequential-selection:{trainer.step}"),
                device=device,
            )
            rank = selection_rank(selection)
            _append_jsonl(
                metrics_path,
                {"type": "checkpoint_selection", "step": trainer.step, **selection},
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                save_named("best.pt", selection)
        if trainer.step % checkpoint_interval == 0:
            save_named("latest.pt", selection)

    save_named("latest.pt")
    print(
        f"model_kind=sequential_conditional steps={trainer.step} run={run_directory}"
    )


if __name__ == "__main__":
    main()
