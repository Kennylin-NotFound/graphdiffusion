"""Run the bounded pre-training acceptance checks for the Stage 3 model family."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.data import create_toy_instance
from gdm_factor_diffusion.diffusion import (
    AbsorbingMaskSchedule,
    corrupt_with_absorbing_mask,
)
from gdm_factor_diffusion.graph import build_factor_graph_batch, infer_feature_schema
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    solve_with_masked_model,
)
from gdm_factor_diffusion.models import (
    ConditionalDenoiserConfig,
    TypedFactorConditionalDenoiser,
)
from gdm_factor_diffusion.solver import enumerate_feasible_placements, verify_placement
from gdm_factor_diffusion.training import (
    MaskedConditionalTrainer,
    compute_masked_objective,
    restore_masked_checkpoint,
    save_masked_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive.")
    torch.manual_seed(args.seed)
    target_device = _device(args.device)
    generator = torch.Generator(device=target_device).manual_seed(args.seed + 1)
    instance = create_toy_instance()
    exhaustive = enumerate_feasible_placements(instance)
    clean_cpu = torch.from_numpy(exhaustive.placements[0][None, :]).long()
    batch = build_factor_graph_batch([instance]).to(target_device)
    clean = clean_cpu.to(target_device)
    schedule = AbsorbingMaskSchedule(num_steps=4)
    model_config = ConditionalDenoiserConfig(
        num_mask_steps=schedule.num_steps,
        hidden_dim=48,
        num_layers=2,
    )
    model = TypedFactorConditionalDenoiser.from_batch(batch, model_config).to(
        target_device
    )
    trainer = MaskedConditionalTrainer(
        model,
        schedule,
        torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5),
    )

    first_loss = None
    final_metrics: dict[str, float] = {}
    for step in range(args.steps):
        timestep = torch.tensor(
            [step % schedule.num_steps + 1],
            dtype=torch.long,
            device=target_device,
        )
        final_metrics = trainer.train_step(
            batch,
            clean,
            timestep,
            generator=generator,
        )
        if first_loss is None:
            first_loss = final_metrics["loss_total"]

    model.eval()
    accuracy_by_timestep: dict[str, float] = {}
    with torch.no_grad():
        for timestep in range(1, schedule.num_steps + 1):
            evaluation_generator = torch.Generator(device=target_device).manual_seed(
                args.seed + 100 + timestep
            )
            partial = corrupt_with_absorbing_mask(
                clean,
                timestep,
                batch.candidate_mask,
                schedule,
                batch.service_mask,
                generator=evaluation_generator,
            )
            logits = model(batch, partial, timestep)
            terms = compute_masked_objective(logits, clean, partial, batch)
            accuracy_by_timestep[str(timestep)] = float(terms.hidden_accuracy.item())
    minimum_accuracy = min(accuracy_by_timestep.values())

    implementation_root = Path(__file__).resolve().parents[1]
    artifact_root = implementation_root / "artifacts" / "phase6e-e-stage3-pretraining"
    checkpoint_path = save_masked_checkpoint(
        artifact_root / "toy_overfit_checkpoint.pt",
        trainer,
        metadata={
            "purpose": "pretraining_acceptance_only",
            "seed": args.seed,
            "steps": args.steps,
            "model_config": {
                "num_mask_steps": model_config.num_mask_steps,
                "hidden_dim": model_config.hidden_dim,
                "num_layers": model_config.num_layers,
                "dropout": model_config.dropout,
            },
        },
    )
    restored_model = TypedFactorConditionalDenoiser.from_batch(
        batch, model_config
    ).to(target_device)
    restored_trainer = MaskedConditionalTrainer(
        restored_model,
        schedule,
        torch.optim.AdamW(restored_model.parameters(), lr=3e-3, weight_decay=1e-5),
    )
    restored_payload = restore_masked_checkpoint(
        checkpoint_path, restored_trainer, map_location=target_device
    )
    with torch.no_grad():
        all_hidden = corrupt_with_absorbing_mask(
            clean, schedule.num_steps, batch.candidate_mask, schedule
        )
        original_logits = model(batch, all_hidden, schedule.num_steps)
        restored_logits = restored_model(batch, all_hidden, schedule.num_steps)
    checkpoint_exact = bool(torch.equal(original_logits, restored_logits))

    schema = infer_feature_schema([instance])
    deterministic = solve_with_masked_model(
        model,
        instance,
        schedule,
        schema,
        decode_config=MaskedDecodeConfig(
            num_samples=1, sample_batch_size=1, stochastic=False
        ),
        inference_config=InferenceConfig(
            num_samples=1,
            sample_batch_size=1,
            enable_repair=False,
            enable_fallback=False,
        ),
        device=target_device,
    )
    stochastic = solve_with_masked_model(
        model,
        instance,
        schedule,
        schema,
        decode_config=MaskedDecodeConfig(
            num_samples=8,
            sample_batch_size=8,
            stochastic=True,
            temperature=0.25,
        ),
        inference_config=InferenceConfig(
            num_samples=8,
            sample_batch_size=8,
            enable_repair=True,
            enable_fallback=True,
        ),
        device=target_device,
        generator=torch.Generator(device=target_device).manual_seed(args.seed + 2),
    )
    deterministic_verified = bool(
        deterministic.success
        and verify_placement(instance, deterministic.placement).feasible
    )
    stochastic_verified = bool(
        stochastic.success and verify_placement(instance, stochastic.placement).feasible
    )
    acceptance = {
        "minimum_hidden_accuracy_at_least_0_99": minimum_accuracy >= 0.99,
        "loss_decreased": bool(final_metrics["loss_total"] < float(first_loss)),
        "checkpoint_resume_exact": checkpoint_exact,
        "deterministic_decode_verified": deterministic_verified,
        "stochastic_decode_verified": stochastic_verified,
        "finite_gradient": bool(np.isfinite(final_metrics["gradient_norm"])),
    }
    report = {
        "schema_version": "1.0",
        "purpose": "phase6e_e_stage3_pretraining_acceptance",
        "device": str(target_device),
        "seed": args.seed,
        "steps": args.steps,
        "first_loss": first_loss,
        "final_metrics": final_metrics,
        "hidden_accuracy_by_timestep": accuracy_by_timestep,
        "minimum_hidden_accuracy": minimum_accuracy,
        "checkpoint_path": str(checkpoint_path),
        "deterministic_metrics": deterministic.metrics,
        "stochastic_metrics": stochastic.metrics,
        "acceptance": acceptance,
        "passed": all(acceptance.values()),
    }
    report_path = write_json(artifact_root / "mvp_acceptance.json", report)
    print(
        f"device={target_device} steps={args.steps} "
        f"first_loss={first_loss:.6f} final_loss={final_metrics['loss_total']:.6f} "
        f"min_hidden_accuracy={minimum_accuracy:.3f} passed={report['passed']}"
    )
    print(f"report={report_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
