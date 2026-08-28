"""Run resumable factor-denoiser training with constrained checkpoint selection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import (
    collect_run_metadata,
    configure_logging,
    create_run_directory,
    write_json,
)
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.diffusion import CategoricalSchedule
from gdm_factor_diffusion.graph import merge_feature_schemas
from gdm_factor_diffusion.models import (
    DenoiserConfig,
    DirectPredictorConfig,
    TypedFactorDenoiser,
    TypedFactorDirectPredictor,
)
from gdm_factor_diffusion.training import (
    ConstrainedValidationConfig,
    DenoiserTrainer,
    LabeledDeploymentDataset,
    TrainerConfig,
    audit_dataset_freeze,
    capture_random_state,
    evaluate_constrained_validation,
    make_labeled_collator,
    restore_checkpoint,
    restore_random_state,
    sample_clean_targets,
    sample_training_batch,
    save_checkpoint,
    validation_rank,
)


_RESUME_MUTABLE_KEYS = {
    "steps",
    "log_interval",
    "denoising_validation_interval",
    "constrained_validation_interval",
    "checkpoint_interval",
    "run_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable dotted.path=value configuration override.",
    )
    return parser.parse_args()


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _feature_schema_payload(feature_schema) -> dict[str, list[str]]:
    return {
        "service_feature_names": list(feature_schema.service_feature_names),
        "device_feature_names": list(feature_schema.device_feature_names),
        "resource_names": list(feature_schema.resource_names),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_signature(config: dict[str, Any]) -> dict[str, Any]:
    signature = copy.deepcopy(config)
    training = signature.get("training", {})
    for key in _RESUME_MUTABLE_KEYS:
        training.pop(key, None)
    return signature


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _constrained_config(payload: dict[str, Any]) -> ConstrainedValidationConfig:
    return ConstrainedValidationConfig(
        num_samples=int(payload["num_samples"]),
        sample_batch_size=int(payload["sample_batch_size"]),
        reverse_steps=(
            None
            if payload.get("reverse_steps") is None
            else int(payload["reverse_steps"])
        ),
        repair_max_moves=int(payload["repair_max_moves"]),
        fallback_max_search_nodes=int(payload["fallback_max_search_nodes"]),
        instance_limit=(
            None
            if payload.get("instance_limit") is None
            else int(payload["instance_limit"])
        ),
    )


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (
        implementation_root / "configs" / "training_phase5a_acceptance.yaml"
    )
    config = load_config(config_path, overrides=args.override)
    training = config["training"]
    diffusion = config["diffusion"]
    model_config = config["model"]
    model_kind = str(training.get("model_kind", "diffusion"))
    if model_kind not in {"diffusion", "direct"}:
        raise ValueError("training.model_kind must be 'diffusion' or 'direct'.")
    training["model_kind"] = model_kind
    seed = int(training["seed"])
    seed_everything(seed)

    dataset_root = _resolve(implementation_root, training["dataset_root"])
    require_freeze = bool(training.get("require_dataset_freeze", False))
    if require_freeze:
        freeze = audit_dataset_freeze(dataset_root)
        training["resolved_dataset_freeze_sha256"] = _sha256(
            dataset_root / "dataset_freeze.json"
        )
        training["resolved_dataset_core_sha256"] = freeze["core_sha256"]
    train_dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=tuple(training["train_partitions"]),
        require_freeze=require_freeze,
    )
    validation_dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=tuple(training["validation_partitions"]),
        require_freeze=require_freeze,
    )
    feature_schema = merge_feature_schemas(
        (train_dataset.feature_schema, validation_dataset.feature_schema)
    )
    collate = make_labeled_collator(feature_schema)
    validation_batch = collate(
        [validation_dataset[index] for index in range(len(validation_dataset))]
    )
    reference = collate([train_dataset[0]]).factor_graph
    device = torch.device(
        training["device"] if torch.cuda.is_available() else "cpu"
    )
    reference = reference.to(device)
    if model_kind == "diffusion":
        model = TypedFactorDenoiser.from_batch(
            reference,
            DenoiserConfig(
                num_diffusion_steps=int(diffusion["steps"]),
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["layers"]),
                dropout=float(model_config["dropout"]),
            ),
        ).to(device)
    else:
        model = TypedFactorDirectPredictor.from_batch(
            reference,
            DirectPredictorConfig(
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["layers"]),
                dropout=float(model_config["dropout"]),
            ),
        ).to(device)
    schedule = CategoricalSchedule.linear(
        int(diffusion["steps"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    trainer = DenoiserTrainer(
        model,
        schedule,
        optimizer,
        TrainerConfig(
            capacity_weight=float(training["capacity_weight"]),
            link_weight=float(training["link_weight"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        ),
        model_kind=model_kind,
    )
    streams = {
        "batch": torch.Generator().manual_seed(derive_seed(seed, "batch")),
        "target": torch.Generator().manual_seed(derive_seed(seed, "target")),
    }
    if model_kind == "diffusion":
        streams["diffusion"] = torch.Generator(device=device).manual_seed(
            derive_seed(seed, "diffusion")
        )

    selection_state: dict[str, Any] = {
        "best_rank": None,
        "best_step": None,
        "best_constrained_metrics": None,
        "best_denoising_loss": None,
    }
    run_root = args.output or implementation_root / "artifacts" / "runs"
    if args.resume is None:
        run_directory = create_run_directory(run_root, str(training["run_name"]))
        write_json(run_directory / "config.json", config)
        write_json(
            run_directory / "run_meta.json",
            collect_run_metadata(seed, config, project_root=implementation_root.parent),
        )
    else:
        checkpoint_path = args.resume.resolve()
        run_directory = checkpoint_path.parent
        payload = restore_checkpoint(checkpoint_path, trainer, map_location=device)
        saved_config = payload["metadata"]["config"]
        if _resume_signature(saved_config) != _resume_signature(config):
            raise ValueError("Resume checkpoint and requested training config disagree.")
        restore_random_state(payload["runtime_state"], streams)
        selection_state.update(payload["metadata"].get("selection_state", {}))
        _append_jsonl(
            run_directory / "resume_events.jsonl",
            collect_run_metadata(seed, config, project_root=implementation_root.parent),
        )

    logger = configure_logging(run_directory)
    metrics_path = run_directory / "metrics.jsonl"
    total_steps = int(training["steps"])
    if trainer.step >= total_steps:
        raise ValueError("Checkpoint step already meets or exceeds requested steps.")
    constrained_settings = _constrained_config(config["constrained_validation"])
    latest_denoising: dict[str, Any] | None = None
    termination_reason = "completed"

    def checkpoint_metadata() -> dict[str, Any]:
        return {
            "config": config,
            "model_kind": model_kind,
            "feature_schema": _feature_schema_payload(feature_schema),
            "selection_state": selection_state,
            "latest_denoising_validation": latest_denoising,
            "termination_reason": termination_reason,
        }

    def save_named_checkpoint(name: str) -> Path:
        return save_checkpoint(
            run_directory / name,
            trainer,
            metadata=checkpoint_metadata(),
            runtime_state=capture_random_state(streams),
        )

    def run_denoising_validation() -> dict[str, Any]:
        target = sample_clean_targets(validation_batch, mode="best")
        graph = validation_batch.factor_graph.to(device)
        timestep = (
            torch.full(
                (graph.batch_size,),
                schedule.num_steps,
                dtype=torch.long,
                device=device,
            )
            if model_kind == "diffusion"
            else None
        )
        generator = (
            torch.Generator(device=device).manual_seed(
                derive_seed(seed, f"denoising-validation:{trainer.step}")
            )
            if model_kind == "diffusion"
            else None
        )
        terms = trainer.evaluate_step(
            graph,
            target.state.to(device),
            timestep,
            generator=generator,
        )
        result = terms.detached_metrics()
        result.update(
            {
                "step": trainer.step,
                "split": "validation_denoising",
                "model_kind": model_kind,
                "validation_role": "clean_state_prediction",
            }
        )
        _append_jsonl(metrics_path, result)
        return result

    try:
        while trainer.step < total_steps:
            labeled_batch = sample_training_batch(
                train_dataset,
                collate,
                batch_size=int(training["batch_size"]),
                generator=streams["batch"],
            )
            target = sample_clean_targets(
                labeled_batch,
                mode=str(training["target_sampling"]),
                generator=streams["target"],
            )
            graph = labeled_batch.factor_graph.to(device)
            timestep = (
                torch.randint(
                    1,
                    schedule.num_steps + 1,
                    (graph.batch_size,),
                    device=device,
                    generator=streams["diffusion"],
                )
                if model_kind == "diffusion"
                else None
            )
            metrics = trainer.train_step(
                graph,
                target.state.to(device),
                timestep,
                generator=streams.get("diffusion"),
            )
            metrics.update(
                {
                    "split": "train",
                    "model_kind": model_kind,
                    "sampled_latency_mean": float(target.latency.mean().item()),
                    "sampled_energy_mean": float(
                        target.normalized_energy.mean().item()
                    ),
                }
            )
            if trainer.step == 1 or trainer.step % int(training["log_interval"]) == 0:
                _append_jsonl(metrics_path, metrics)

            denoising_due = (
                trainer.step == 1
                or trainer.step % int(training["denoising_validation_interval"]) == 0
                or trainer.step == total_steps
            )
            if denoising_due:
                latest_denoising = run_denoising_validation()

            constrained_due = (
                trainer.step % int(training["constrained_validation_interval"]) == 0
                or trainer.step == total_steps
            )
            if constrained_due:
                if latest_denoising is None or latest_denoising["step"] != trainer.step:
                    latest_denoising = run_denoising_validation()
                constrained = evaluate_constrained_validation(
                    model,
                    schedule,
                    feature_schema,
                    validation_dataset,
                    config=constrained_settings,
                    device=device,
                    seed=derive_seed(seed, "constrained-validation"),
                    model_kind=model_kind,
                )
                constrained.update(
                    {"step": trainer.step, "split": "validation_constrained"}
                )
                _append_jsonl(metrics_path, constrained)
                rank = validation_rank(
                    constrained, float(latest_denoising["loss_total"])
                )
                best_rank = selection_state["best_rank"]
                if best_rank is None or rank < tuple(best_rank):
                    selection_state.update(
                        {
                            "best_rank": list(rank),
                            "best_step": trainer.step,
                            "best_constrained_metrics": constrained,
                            "best_denoising_loss": latest_denoising["loss_total"],
                        }
                    )
                    save_named_checkpoint("best_checkpoint.pt")
                logger.info(
                    "step=%d verified=%.4f gap=%s val_loss=%.6f",
                    trainer.step,
                    constrained["verified_rate"],
                    constrained["mean_gap_to_pool_best"],
                    latest_denoising["loss_total"],
                )

            if trainer.step % int(training["checkpoint_interval"]) == 0:
                save_named_checkpoint("latest_checkpoint.pt")
    except KeyboardInterrupt:
        termination_reason = "interrupted"
        logger.warning("Training interrupted at completed step %d.", trainer.step)
    except Exception:
        termination_reason = "failed"
        logger.exception("Training failed at completed step %d.", trainer.step)
        raise
    finally:
        save_named_checkpoint("latest_checkpoint.pt")
        save_named_checkpoint("final_checkpoint.pt")
        summary = {
            "run_directory": str(run_directory),
            "termination_reason": termination_reason,
            "completed_steps": trainer.step,
            "requested_steps": total_steps,
            "selection_state": selection_state,
            "final_checkpoint": str(run_directory / "final_checkpoint.pt"),
        }
        write_json(run_directory / "summary.json", summary)
        logger.info(
            "training finished reason=%s step=%d best_step=%s",
            termination_reason,
            trainer.step,
            selection_state["best_step"],
        )


if __name__ == "__main__":
    main()
