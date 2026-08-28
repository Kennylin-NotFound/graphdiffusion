"""Preflight or train the Stage 3 masked model and its direct control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.diffusion import AbsorbingMaskSchedule, CategoricalSchedule
from gdm_factor_diffusion.graph import merge_feature_schemas
from gdm_factor_diffusion.models import (
    ConditionalDenoiserConfig,
    DirectPredictorConfig,
    TypedFactorConditionalDenoiser,
    TypedFactorDirectPredictor,
)
from gdm_factor_diffusion.training import (
    DenoiserTrainer,
    LabeledDeploymentDataset,
    MaskedConditionalTrainer,
    MaskedTrainerConfig,
    Stage3SelectionConfig,
    TrainerConfig,
    audit_dataset_freeze,
    capture_random_state,
    evaluate_stage3_selection,
    make_labeled_collator,
    restore_checkpoint,
    restore_masked_checkpoint,
    restore_random_state,
    sample_clean_targets,
    sample_training_batch,
    save_checkpoint,
    save_masked_checkpoint,
    stage3_selection_rank,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Stage 3 training YAML; defaults to the locked pilot config.",
    )
    parser.add_argument(
        "--model-kind",
        choices=("masked_conditional", "direct"),
        default="masked_conditional",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate data/model/inference contracts without an optimizer step.",
    )
    return parser.parse_args()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _selection_config(config: dict[str, Any]) -> Stage3SelectionConfig:
    value = config["checkpoint_selection"]
    return Stage3SelectionConfig(
        instance_limit=int(value["instance_limit"]),
        sample_batch_size=int(value["sample_batch_size"]),
        temperature=float(value["temperature"]),
        repair_max_moves=int(value["repair_max_moves"]),
        fallback_max_search_nodes=int(value["fallback_max_search_nodes"]),
    )


def main() -> None:
    if os.environ.get("GDM_STAGE3_ACTIVE_ENTRY") != "1":
        raise RuntimeError(
            "Direct Stage 3 trainer invocation is disabled. "
            "Use implementation/active_stage3/run_training.ps1."
        )
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (
        implementation_root
        / "configs"
        / "training_phase6e_e_stage3_pilot.yaml"
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
    optimization = config["optimization"]
    mask_schedule: AbsorbingMaskSchedule | None = None

    if args.model_kind == "masked_conditional":
        model_config = config["model"]
        mask_schedule = AbsorbingMaskSchedule(
            num_steps=int(model_config["num_mask_steps"]),
            power=float(model_config["mask_schedule_power"]),
        )
        model = TypedFactorConditionalDenoiser.from_batch(
            reference,
            ConditionalDenoiserConfig(
                num_mask_steps=mask_schedule.num_steps,
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["num_layers"]),
                dropout=float(model_config["dropout"]),
            ),
        ).to(device)
    else:
        model_config = config["direct_control"]
        model = TypedFactorDirectPredictor.from_batch(
            reference,
            DirectPredictorConfig(
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["num_layers"]),
                dropout=float(model_config["dropout"]),
            ),
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    if args.model_kind == "masked_conditional":
        assert mask_schedule is not None
        trainer: Any = MaskedConditionalTrainer(
            model,
            mask_schedule,
            optimizer,
            MaskedTrainerConfig(
                gradient_clip_norm=float(optimization["gradient_clip_norm"])
            ),
        )
    else:
        trainer = DenoiserTrainer(
            model,
            CategoricalSchedule.linear(1, beta_start=0.1, beta_end=0.1),
            optimizer,
            TrainerConfig(
                capacity_weight=0.0,
                link_weight=0.0,
                gradient_clip_norm=float(optimization["gradient_clip_norm"]),
            ),
            model_kind="direct",
        )

    streams = {
        "batch": torch.Generator().manual_seed(derive_seed(seed, "stage3-batch")),
        "target": torch.Generator().manual_seed(derive_seed(seed, "stage3-target")),
    }
    if args.model_kind == "masked_conditional":
        streams["mask"] = torch.Generator(device=device).manual_seed(
            derive_seed(seed, "stage3-mask")
        )

    if args.preflight:
        batch = collate([train_dataset[0], train_dataset[1]])
        target = sample_clean_targets(batch, mode="best")
        graph = batch.factor_graph.to(device)
        if args.model_kind == "masked_conditional":
            assert mask_schedule is not None
            terms = trainer.evaluate_step(
                graph,
                target.state.to(device),
                torch.full(
                    (graph.batch_size,),
                    mask_schedule.num_steps,
                    dtype=torch.long,
                    device=device,
                ),
                generator=streams["mask"],
            )
            loss = float(terms.total.item())
        else:
            terms = trainer.evaluate_step(graph, target.state.to(device))
            loss = float(terms.total.item())
        selection_config = _selection_config(config)
        selection_config = Stage3SelectionConfig(
            instance_limit=1,
            sample_batch_size=selection_config.sample_batch_size,
            temperature=selection_config.temperature,
            repair_max_moves=selection_config.repair_max_moves,
            fallback_max_search_nodes=selection_config.fallback_max_search_nodes,
        )
        selection = evaluate_stage3_selection(
            model,
            feature_schema,
            checkpoint_dataset,
            model_kind=args.model_kind,
            config=selection_config,
            seed=seed,
            device=device,
            mask_schedule=mask_schedule,
        )
        report = {
            "schema_version": "1.0",
            "mode": "preflight_no_optimizer_step",
            "model_kind": args.model_kind,
            "device": str(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "dataset_freeze": freeze["core_sha256"],
            "batch_loss": loss,
            "selection_smoke": selection,
            "passed": torch.isfinite(torch.tensor(loss)).item(),
        }
        destination = (
            implementation_root
            / "artifacts"
            / "phase6e-e-stage3"
            / f"preflight_{args.model_kind}.json"
        )
        write_json(destination, report)
        print(
            f"model_kind={args.model_kind} mode=preflight device={device} "
            f"loss={loss:.6f} passed={report['passed']} report={destination}"
        )
        return

    output = config["output"]
    run_name = output["run_pattern"].format(model_kind=args.model_kind)
    run_directory = _resolve(implementation_root, output["root"]) / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = run_directory / output["metrics"]
    best_rank: tuple[float, ...] | None = None

    def metadata(selection: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "config": config,
            "model_kind": args.model_kind,
            "feature_schema": {
                "service_feature_names": list(feature_schema.service_feature_names),
                "device_feature_names": list(feature_schema.device_feature_names),
                "resource_names": list(feature_schema.resource_names),
            },
            "best_rank": best_rank,
            "latest_selection": selection,
        }

    def save_named(name: str, selection: dict[str, Any] | None = None) -> Path:
        path = run_directory / name
        runtime = capture_random_state(streams)
        if args.model_kind == "masked_conditional":
            return save_masked_checkpoint(
                path, trainer, metadata=metadata(selection), runtime_state=runtime
            )
        return save_checkpoint(
            path, trainer, metadata=metadata(selection), runtime_state=runtime
        )

    if args.resume is not None:
        if args.model_kind == "masked_conditional":
            payload = restore_masked_checkpoint(
                args.resume, trainer, map_location=device
            )
        else:
            payload = restore_checkpoint(args.resume, trainer, map_location=device)
        restore_random_state(payload["runtime_state"], streams)
        saved_rank = payload["metadata"].get("best_rank")
        best_rank = None if saved_rank is None else tuple(saved_rank)
    elif metrics_path.exists():
        raise FileExistsError(
            f"Run already exists; pass --resume to continue: {run_directory}"
        )
    write_json(run_directory / "config.json", config)

    max_steps = int(optimization["max_steps"])
    validation_interval = int(optimization["validation_interval"])
    checkpoint_interval = int(optimization["checkpoint_interval"])
    selection_config = _selection_config(config)
    while trainer.step < max_steps:
        batch = sample_training_batch(
            train_dataset,
            collate,
            batch_size=int(optimization["batch_size"]),
            generator=streams["batch"],
        )
        target = sample_clean_targets(batch, mode="best", generator=streams["target"])
        graph = batch.factor_graph.to(device)
        if args.model_kind == "masked_conditional":
            metrics = trainer.train_step(
                graph,
                target.state.to(device),
                generator=streams["mask"],
            )
        else:
            metrics = trainer.train_step(graph, target.state.to(device))
        if trainer.step == 1 or trainer.step % 50 == 0:
            _append_jsonl(metrics_path, {"type": "train", **metrics})

        selection = None
        if trainer.step % validation_interval == 0:
            selection = evaluate_stage3_selection(
                model,
                feature_schema,
                checkpoint_dataset,
                model_kind=args.model_kind,
                config=selection_config,
                seed=derive_seed(seed, f"selection:{trainer.step}"),
                device=device,
                mask_schedule=mask_schedule,
            )
            rank = stage3_selection_rank(selection)
            _append_jsonl(
                metrics_path,
                {"type": "checkpoint_selection", "step": trainer.step, **selection},
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                save_named(output["best_checkpoint"], selection)
        if trainer.step % checkpoint_interval == 0:
            save_named(output["latest_checkpoint"], selection)

    save_named(output["latest_checkpoint"])
    print(
        f"model_kind={args.model_kind} steps={trainer.step} run={run_directory}"
    )


if __name__ == "__main__":
    main()
