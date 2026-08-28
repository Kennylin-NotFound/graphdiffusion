"""Audit soft-guidance activation and random categorical feasibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.data import load_manifest, load_manifest_instance
from gdm_factor_diffusion.diffusion import masked_uniform, sample_prior
from gdm_factor_diffusion.graph import build_factor_graph_batch
from gdm_factor_diffusion.solver import verify_placement
from gdm_factor_diffusion.training import capacity_guidance, link_guidance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=None,
        help="Labeled dataset root.",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive.")
    implementation_root = Path(__file__).resolve().parents[1]
    root = args.dataset_root or (
        implementation_root / "artifacts" / "datasets" / "phase1b-smoke"
    )
    manifest = load_manifest(root / "manifest.json")
    records: list[dict[str, float | str]] = []
    for entry in manifest["instances"]:
        instance = load_manifest_instance(root, entry)
        batch = build_factor_graph_batch([instance])
        uniform = masked_uniform(batch.candidate_mask, batch.service_mask)
        soft_capacity = float(capacity_guidance(uniform, batch).item())
        soft_link = float(link_guidance(uniform, batch).item())
        generator = torch.Generator().manual_seed(int(entry["seed"]))
        capacity_violations = 0
        link_violations = 0
        feasible = 0
        for _ in range(args.samples):
            placement = sample_prior(
                batch.candidate_mask,
                batch.service_mask,
                generator=generator,
            )[0].numpy()
            verification = verify_placement(instance, placement)
            capacity_violations += int(not verification.capacity_valid)
            link_violations += int(not verification.direct_link_valid)
            feasible += int(verification.feasible)
        records.append(
            {
                "instance_id": instance.instance_id,
                "partition": entry["partition"],
                "uniform_soft_capacity": soft_capacity,
                "uniform_soft_link": soft_link,
                "random_capacity_violation_rate": capacity_violations / args.samples,
                "random_link_violation_rate": link_violations / args.samples,
                "random_feasible_rate": feasible / args.samples,
            }
        )
    summary = {
        "dataset_name": manifest["dataset_name"],
        "samples_per_instance": args.samples,
        "mean_uniform_soft_capacity": float(
            np.mean([record["uniform_soft_capacity"] for record in records])
        ),
        "mean_uniform_soft_link": float(
            np.mean([record["uniform_soft_link"] for record in records])
        ),
        "mean_random_capacity_violation_rate": float(
            np.mean(
                [record["random_capacity_violation_rate"] for record in records]
            )
        ),
        "mean_random_link_violation_rate": float(
            np.mean([record["random_link_violation_rate"] for record in records])
        ),
        "mean_random_feasible_rate": float(
            np.mean([record["random_feasible_rate"] for record in records])
        ),
        "records": records,
    }
    output = args.output or (
        implementation_root
        / "artifacts"
        / "audits"
        / "phase4a_training_guidance.json"
    )
    write_json(output, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
