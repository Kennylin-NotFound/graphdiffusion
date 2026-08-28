"""Run a short audited Phase 4A energy-weighted training smoke experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import create_run_directory, write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.diffusion import CategoricalSchedule
from gdm_factor_diffusion.graph import merge_feature_schemas
from gdm_factor_diffusion.models import DenoiserConfig, TypedFactorDenoiser
from gdm_factor_diffusion.training import (
    DenoiserTrainer,
    LabeledDeploymentDataset,
    TrainerConfig,
    make_labeled_collator,
    sample_clean_targets,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (
        implementation_root / "configs" / "training_phase4a_smoke.yaml"
    )
    config = load_config(config_path)
    training = config["training"]
    diffusion = config["diffusion"]
    model_config = config["model"]
    seed = int(training["seed"])
    seed_everything(seed)

    configured_root = Path(training["dataset_root"])
    dataset_root = (
        configured_root
        if configured_root.is_absolute()
        else implementation_root / configured_root
    )
    train_dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=tuple(training["train_partitions"]),
    )
    validation_dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=tuple(training["validation_partitions"]),
    )
    feature_schema = merge_feature_schemas(
        (train_dataset.feature_schema, validation_dataset.feature_schema)
    )
    collate = make_labeled_collator(feature_schema)
    loader_generator = torch.Generator().manual_seed(derive_seed(seed, "loader"))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=len(validation_dataset),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    reference = next(iter(train_loader)).factor_graph
    device = torch.device(
        training["device"] if torch.cuda.is_available() else "cpu"
    )
    reference = reference.to(device)
    model = TypedFactorDenoiser.from_batch(
        reference,
        DenoiserConfig(
            num_diffusion_steps=int(diffusion["steps"]),
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
    )

    run_root = args.output or (
        implementation_root / "artifacts" / "runs"
    )
    run_directory = create_run_directory(run_root, "phase4a-smoke")
    write_json(run_directory / "config.json", config)
    cpu_generator = torch.Generator().manual_seed(derive_seed(seed, "target"))
    cuda_generator = torch.Generator(device=device).manual_seed(
        derive_seed(seed, "diffusion")
    )
    train_iterator = iter(train_loader)
    history: list[dict[str, float | int | str]] = []
    best_validation_loss = float("inf")

    for step in range(1, int(training["steps"]) + 1):
        labeled_batch, train_iterator = _next_batch(train_iterator, train_loader)
        target = sample_clean_targets(
            labeled_batch,
            mode=str(training["target_sampling"]),
            generator=cpu_generator,
        )
        factor_graph = labeled_batch.factor_graph.to(device)
        clean_state = target.state.to(device)
        timestep = torch.randint(
            1,
            schedule.num_steps + 1,
            (factor_graph.batch_size,),
            device=device,
            generator=cuda_generator,
        )
        metrics = trainer.train_step(
            factor_graph,
            clean_state,
            timestep,
            generator=cuda_generator,
        )
        metrics["sampled_latency_mean"] = float(target.latency.mean().item())
        metrics["sampled_energy_mean"] = float(
            target.normalized_energy.mean().item()
        )
        metrics["split"] = "train"
        history.append(metrics)

        if step == 1 or step % int(training["validation_interval"]) == 0:
            validation_batch = next(iter(validation_loader))
            validation_target = sample_clean_targets(
                validation_batch,
                mode="best",
            )
            validation_graph = validation_batch.factor_graph.to(device)
            validation_state = validation_target.state.to(device)
            validation_timestep = torch.full(
                (validation_graph.batch_size,),
                schedule.num_steps,
                dtype=torch.long,
                device=device,
            )
            validation_generator = torch.Generator(device=device).manual_seed(
                derive_seed(seed, f"validation:{step}")
            )
            terms = trainer.evaluate_step(
                validation_graph,
                validation_state,
                validation_timestep,
                generator=validation_generator,
            )
            validation_metrics = terms.detached_metrics()
            validation_metrics.update({"step": step, "split": "validation"})
            history.append(validation_metrics)
            print(
                f"step={step} train_loss={metrics['loss_total']:.6f} "
                f"train_acc={metrics['clean_accuracy']:.4f} "
                f"val_loss={validation_metrics['loss_total']:.6f} "
                f"val_acc={validation_metrics['clean_accuracy']:.4f}"
            )
            if validation_metrics["loss_total"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss_total"]
                save_checkpoint(
                    run_directory / "best_checkpoint.pt",
                    trainer,
                    metadata={
                        "config": config,
                        "feature_schema": {
                            "service_feature_names": list(
                                feature_schema.service_feature_names
                            ),
                            "device_feature_names": list(
                                feature_schema.device_feature_names
                            ),
                            "resource_names": list(feature_schema.resource_names),
                        },
                        "best_validation_loss": best_validation_loss,
                    },
                )

    with (run_directory / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for record in history:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "run_directory": str(run_directory),
        "steps": trainer.step,
        "best_validation_loss": best_validation_loss,
        "final_train_metrics": history[-2] if len(history) >= 2 else history[-1],
        "final_validation_metrics": history[-1],
    }
    write_json(run_directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
