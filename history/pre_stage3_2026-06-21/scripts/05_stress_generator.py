"""Stress-test constructive generation across broad parameter ranges."""

from __future__ import annotations

import argparse

import numpy as np

from gdm_factor_diffusion.data import (
    InstanceGenerationSpec,
    audit_graph_readiness,
    generate_instance,
)
from gdm_factor_diffusion.solver import evaluate_latency, verify_placement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    service_counts: list[int] = []
    candidate_counts: list[int] = []
    objectives: list[float] = []

    for index in range(args.count):
        spec = InstanceGenerationSpec(
            instance_id=f"stress-{index:05d}",
            seed=int(rng.integers(0, 2**32 - 1)),
            partition="stress",
            role="audit",
            regime="stress",
            size_profile="mixed",
            num_applications=int(rng.integers(1, 7)),
            num_devices=int(rng.integers(1, 11)),
            share_probability=float(rng.uniform(0, 1)),
            compatibility_density=float(rng.uniform(0, 1)),
            topology_density=float(rng.uniform(0, 1)),
            capacity_slack=float(rng.uniform(0, 0.8)),
            minimum_candidates=int(rng.integers(1, 4)),
        )
        generated = generate_instance(spec)
        instance = generated.instance
        if not verify_placement(instance, generated.witness_placement).feasible:
            raise RuntimeError(f"Witness verification failed for {spec.instance_id}.")
        audit = audit_graph_readiness(instance, generated.witness_placement)
        if not audit.ready:
            raise RuntimeError(f"Graph audit failed for {spec.instance_id}: {audit}")
        objective = evaluate_latency(instance, generated.witness_placement).objective
        service_counts.append(instance.num_services)
        candidate_counts.append(int(instance.compatibility_mask.sum()))
        objectives.append(objective)

    print(
        f"generated={args.count} "
        f"services=[{min(service_counts)},{max(service_counts)}] "
        f"candidate_edges=[{min(candidate_counts)},{max(candidate_counts)}] "
        f"objective=[{min(objectives):.6g},{max(objectives):.6g}]"
    )


if __name__ == "__main__":
    main()
