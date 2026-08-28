"""Overfit the minimal factor denoiser on the best smoke-pool placements."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from gdm_factor_diffusion.data import load_manifest, load_manifest_instance
from gdm_factor_diffusion.diffusion import (
    CategoricalSchedule,
    masked_softmax,
    q_sample,
    sample_prior,
)
from gdm_factor_diffusion.graph import build_factor_graph_batch
from gdm_factor_diffusion.models import DenoiserConfig, TypedFactorDenoiser
from gdm_factor_diffusion.solver import load_solution_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=None,
        help="Labeled dataset root containing solution_pool_manifest.json.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    root = args.dataset_root or (
        implementation_root / "artifacts" / "datasets" / "phase1b-smoke"
    )
    manifest = load_manifest(root / "manifest.json")
    train_entries = [
        entry for entry in manifest["instances"] if entry["partition"] == "train"
    ]
    instances = [load_manifest_instance(root, entry) for entry in train_entries]
    pools = [
        load_solution_pool(
            root / "solution_pools" / entry["partition"] / f"{entry['instance_id']}.npz"
        )
        for entry in train_entries
    ]
    batch = build_factor_graph_batch(instances)
    clean_state = torch.full(batch.service_mask.shape, -1, dtype=torch.long)
    for index, pool in enumerate(pools):
        clean_state[index, : pool.placements.shape[1]] = torch.from_numpy(
            pool.placements[0]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)
    clean_state = clean_state.to(device)
    schedule = CategoricalSchedule.linear(
        args.diffusion_steps,
        beta_end=0.2,
        device=device,
    )
    model = TypedFactorDenoiser.from_batch(
        batch,
        DenoiserConfig(
            num_diffusion_steps=args.diffusion_steps,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    fixed_prior = sample_prior(batch.candidate_mask, batch.service_mask, generator=generator)
    final_accuracy = 0.0

    for step in range(1, args.steps + 1):
        model.train()
        timestep = torch.randint(
            1,
            args.diffusion_steps + 1,
            (batch.batch_size,),
            device=device,
            generator=generator,
        )
        noisy_state = q_sample(
            clean_state,
            timestep,
            batch.candidate_mask,
            schedule,
            batch.service_mask,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch, noisy_state, timestep)
        loss = F.cross_entropy(
            logits[batch.service_mask],
            clean_state[batch.service_mask],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                validation_logits = model(
                    batch,
                    fixed_prior,
                    args.diffusion_steps,
                )
                probability = masked_softmax(
                    validation_logits,
                    batch.candidate_mask,
                    batch.service_mask,
                )
                prediction = probability.argmax(dim=-1)
                final_accuracy = float(
                    (
                        prediction[batch.service_mask]
                        == clean_state[batch.service_mask]
                    )
                    .float()
                    .mean()
                    .item()
                )
            print(
                f"step={step} loss={float(loss.item()):.6f} "
                f"fixed_prior_clean_accuracy={final_accuracy:.4f}"
            )
            if final_accuracy == 1.0 and float(loss.item()) < 0.02:
                break

    if final_accuracy < 0.98:
        raise RuntimeError(
            f"Factor denoiser failed the toy-overfit gate: accuracy={final_accuracy:.4f}"
        )
    print(
        f"overfit_passed device={device} instances={batch.batch_size} "
        f"accuracy={final_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
