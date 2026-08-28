"""Profile the implemented factor denoiser on the full smoke graph batch."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from gdm_factor_diffusion.data import load_manifest, load_manifest_instance
from gdm_factor_diffusion.diffusion import CategoricalSchedule, q_sample
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
        help="Labeled dataset root.",
    )
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for factor-denoiser profiling.")
    args = parse_args()
    implementation_root = Path(__file__).resolve().parents[1]
    root = args.dataset_root or (
        implementation_root / "artifacts" / "datasets" / "phase1b-smoke"
    )
    manifest = load_manifest(root / "manifest.json")
    instances = [
        load_manifest_instance(root, entry) for entry in manifest["instances"]
    ]
    pools = [
        load_solution_pool(
            root / "solution_pools" / entry["partition"] / f"{entry['instance_id']}.npz"
        )
        for entry in manifest["instances"]
    ]
    batch = build_factor_graph_batch(instances)
    clean_state = torch.full(batch.service_mask.shape, -1, dtype=torch.long)
    for index, pool in enumerate(pools):
        clean_state[index, : pool.placements.shape[1]] = torch.from_numpy(
            pool.placements[0]
        )
    batch = batch.to("cuda")
    clean_state = clean_state.cuda()
    schedule = CategoricalSchedule.linear(100, beta_end=0.2, device="cuda")

    for hidden_dim, layers in ((128, 4), (256, 6)):
        model = TypedFactorDenoiser.from_batch(
            batch,
            DenoiserConfig(
                num_diffusion_steps=100,
                hidden_dim=hidden_dim,
                num_layers=layers,
            ),
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.steps):
            timestep = torch.randint(1, 101, (batch.batch_size,), device="cuda")
            noisy_state = q_sample(
                clean_state,
                timestep,
                batch.candidate_mask,
                schedule,
                batch.service_mask,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch, noisy_state, timestep)
            loss = F.cross_entropy(
                logits[batch.service_mask],
                clean_state[batch.service_mask],
            )
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(
            f"hidden={hidden_dim} layers={layers} "
            f"parameters={sum(parameter.numel() for parameter in model.parameters())} "
            f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.4f} "
            f"milliseconds_per_step={elapsed * 1000 / args.steps:.3f}"
        )
        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
