"""Evaluate Random + Recovery on the realistic-profile dataset.

This script is intentionally separate from the older fallback-based evaluation
pipeline. It tests the manuscript's recovery reference:

1. sample K compatible random proposals;
2. accept verified proposals directly;
3. invoke bounded proposal-conditioned recovery only if no raw proposal is
   verified;
4. rank verified candidates by exact end-to-end latency.

No learned model and no deterministic constructive fallback are used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import numpy as np
import torch

from gdm_factor_diffusion.common.seed import derive_seed
from gdm_factor_diffusion.data import load_manifest, load_manifest_instance
from gdm_factor_diffusion.inference import sample_random_proposals
from gdm_factor_diffusion.inference.repair import RepairConfig, repair_placement, violation_score
from gdm_factor_diffusion.solver import evaluate_latency, verify_placement


DEFAULT_PARTITIONS = ("profile_id", "profile_branched", "profile_high_sharing")
DEFAULT_SEEDS = tuple(range(2026070111, 2026070121))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _parse_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_seeds(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return DEFAULT_SEEDS
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _load_reference_pool_best(records_root: Path) -> dict[str, float]:
    """Load instance reference objectives from existing realistic-profile records."""

    references: dict[str, float] = {}
    for path in records_root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        instance_id = str(payload["instance_id"])
        pool_best = float(payload["pool_best"])
        previous = references.get(instance_id)
        if previous is not None and abs(previous - pool_best) > 1e-9:
            raise ValueError(f"Inconsistent pool_best for {instance_id}: {previous} vs {pool_best}")
        references[instance_id] = pool_best
    if not references:
        raise FileNotFoundError(f"No reference records found under {records_root}")
    return references


def _candidate_key(placement: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in placement.tolist())


def _add_candidate(
    candidates: dict[tuple[int, ...], dict[str, Any]],
    instance: Any,
    placement: np.ndarray,
    *,
    source: str,
    proposal_index: int,
    recovery_moves: int,
) -> None:
    report = verify_placement(instance, placement)
    if not report.feasible:
        return
    objective = float(evaluate_latency(instance, report.placement).objective)
    key = _candidate_key(report.placement)
    candidate = {
        "placement": report.placement.tolist(),
        "objective": objective,
        "source": source,
        "proposal_index": int(proposal_index),
        "recovery_moves": int(recovery_moves),
    }
    previous = candidates.get(key)
    if previous is None or (objective, source, proposal_index) < (
        float(previous["objective"]),
        str(previous["source"]),
        int(previous["proposal_index"]),
    ):
        candidates[key] = candidate


def evaluate_one(
    instance: Any,
    *,
    pool_best: float,
    seed: int,
    num_samples: int,
    recovery_candidate_limit: int,
    recovery_max_moves: int,
) -> dict[str, Any]:
    start = perf_counter()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_seed(seed, f"random_recovery:{instance.instance_id}"))

    sampling_start = perf_counter()
    proposals = sample_random_proposals(
        instance,
        num_samples=num_samples,
        generator=generator,
    )
    sampling_seconds = perf_counter() - sampling_start

    verification_start = perf_counter()
    reports = [verify_placement(instance, proposal) for proposal in proposals]
    verification_seconds = perf_counter() - verification_start

    candidates: dict[tuple[int, ...], dict[str, Any]] = {}
    raw_feasible = 0
    raw_capacity_violations = 0
    raw_link_violations = 0
    evaluation_seconds = 0.0
    for index, report in enumerate(reports):
        raw_feasible += int(report.feasible)
        raw_capacity_violations += int(not report.capacity_valid)
        raw_link_violations += int(not report.direct_link_valid)
        if report.feasible:
            eval_start = perf_counter()
            _add_candidate(
                candidates,
                instance,
                report.placement,
                source="raw",
                proposal_index=index,
                recovery_moves=0,
            )
            evaluation_seconds += perf_counter() - eval_start

    recovery_invoked = not candidates
    recovery_attempts = 0
    recovery_successes = 0
    recovery_moves = 0
    recovery_seconds = 0.0
    if recovery_invoked:
        failed_indices = [
            index for index, report in enumerate(reports) if not report.feasible
        ]
        failed_indices = sorted(
            failed_indices,
            key=lambda index: (violation_score(instance, proposals[index], reports[index]), index),
        )[:recovery_candidate_limit]
        for index in failed_indices:
            recovery_attempts += 1
            recovery_start = perf_counter()
            recovered = repair_placement(
                instance,
                proposals[index],
                config=RepairConfig(max_moves=recovery_max_moves),
            )
            recovery_seconds += perf_counter() - recovery_start
            recovery_moves += len(recovered.moves)
            if recovered.success:
                recovery_successes += 1
                eval_start = perf_counter()
                _add_candidate(
                    candidates,
                    instance,
                    recovered.placement,
                    source="recovery",
                    proposal_index=index,
                    recovery_moves=len(recovered.moves),
                )
                evaluation_seconds += perf_counter() - eval_start

    best = min(candidates.values(), key=lambda item: (float(item["objective"]), str(item["source"]))) if candidates else None
    total_seconds = perf_counter() - start
    objective = None if best is None else float(best["objective"])
    return {
        "seed": int(seed),
        "instance_id": instance.instance_id,
        "partition": instance.metadata.get("partition"),
        "num_services": int(instance.num_services),
        "num_devices": int(instance.num_devices),
        "num_dependencies": int(instance.num_dependencies),
        "num_samples": int(num_samples),
        "success": best is not None,
        "source": None if best is None else str(best["source"]),
        "objective": objective,
        "pool_best": float(pool_best),
        "gap_to_pool_best": None if objective is None else objective / float(pool_best) - 1.0,
        "raw_any_feasible": raw_feasible > 0,
        "raw_feasible_count": int(raw_feasible),
        "raw_feasible_rate": float(raw_feasible / num_samples),
        "raw_capacity_violation_rate": float(raw_capacity_violations / num_samples),
        "raw_link_violation_rate": float(raw_link_violations / num_samples),
        "recovery_invoked": bool(recovery_invoked),
        "recovery_success": bool(recovery_invoked and best is not None),
        "recovery_attempts": int(recovery_attempts),
        "recovery_successes": int(recovery_successes),
        "recovery_success_rate": float(recovery_successes / recovery_attempts) if recovery_attempts else 0.0,
        "recovery_moves": int(recovery_moves),
        "sampling_seconds": float(sampling_seconds),
        "verification_seconds": float(verification_seconds),
        "recovery_seconds": float(recovery_seconds),
        "exact_evaluation_seconds": float(evaluation_seconds),
        "total_seconds": float(total_seconds),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize empty records.")
    successes = [record for record in records if record["success"]]
    gaps = [float(record["gap_to_pool_best"]) for record in successes]
    times = [float(record["total_seconds"]) for record in records]
    summary = {
        "records": len(records),
        "success_rate": sum(record["success"] for record in records) / len(records),
        "raw_any_feasible_rate": sum(record["raw_any_feasible"] for record in records) / len(records),
        "recovery_invocation_rate": sum(record["recovery_invoked"] for record in records) / len(records),
        "recovery_success_rate_over_invoked": (
            sum(record["recovery_success"] for record in records)
            / max(1, sum(record["recovery_invoked"] for record in records))
        ),
        "mean_gap_to_pool_best": None if not gaps else mean(gaps),
        "std_gap_to_pool_best": None if len(gaps) < 2 else pstdev(gaps),
        "mean_total_seconds": mean(times),
        "std_total_seconds": None if len(times) < 2 else pstdev(times),
        "mean_raw_feasible_rate": mean(float(record["raw_feasible_rate"]) for record in records),
        "mean_recovery_attempts": mean(float(record["recovery_attempts"]) for record in records),
        "source_rates": {
            source: sum(record["source"] == source for record in records) / len(records)
            for source in ("raw", "recovery")
        },
    }
    summary["source_rates"]["failure"] = sum(record["source"] is None for record in records) / len(records)
    return summary


def _write_report(output_root: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]

    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{100.0 * float(value):.2f}%"

    def num(value: float | None, digits: int = 3) -> str:
        return "N/A" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# Random + Recovery Realistic-Profile Baseline",
        "",
        "This run samples compatible random proposals, verifies them first, and invokes bounded proposal-conditioned recovery only when no raw proposal is verified. No learned model and no deterministic constructive fallback are used.",
        "",
        "## Overall",
        "",
        "| Method | Records | Succ. | Raw any feasible | Recovery invoked | Recovery success / invoked | Gap (%) | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| Random + Recovery | "
        f"{overall['records']} | {pct(overall['success_rate'])} | "
        f"{pct(overall['raw_any_feasible_rate'])} | "
        f"{pct(overall['recovery_invocation_rate'])} | "
        f"{pct(overall['recovery_success_rate_over_invoked'])} | "
        f"{num(None if overall['mean_gap_to_pool_best'] is None else 100.0 * overall['mean_gap_to_pool_best'])} | "
        f"{num(overall['mean_total_seconds'])} |",
        "",
        "## By Partition",
        "",
        "| Partition | Records | Succ. | Raw any feasible | Recovery invoked | Gap (%) | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for partition, payload in summary["by_partition"].items():
        lines.append(
            f"| {partition} | {payload['records']} | {pct(payload['success_rate'])} | "
            f"{pct(payload['raw_any_feasible_rate'])} | "
            f"{pct(payload['recovery_invocation_rate'])} | "
            f"{num(None if payload['mean_gap_to_pool_best'] is None else 100.0 * payload['mean_gap_to_pool_best'])} | "
            f"{num(payload['mean_total_seconds'])} |"
        )
    lines.extend(["", "## Protocol", "", "```json", json.dumps(summary["protocol"], indent=2, sort_keys=True), "```", ""])
    (output_root / "random_recovery_realistic_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    dataset_root = _resolve(root, args.dataset_root)
    output_root = _resolve(root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    partitions = _parse_csv(args.partitions, DEFAULT_PARTITIONS)
    seeds = _parse_seeds(args.seeds)
    references = _load_reference_pool_best(_resolve(root, args.reference_records_root))
    manifest = load_manifest(dataset_root / "manifest.json")
    entries = [entry for entry in manifest["instances"] if entry["partition"] in set(partitions)]
    if args.max_instances_per_partition is not None:
        limited: list[dict[str, Any]] = []
        for partition in partitions:
            selected = [entry for entry in entries if entry["partition"] == partition]
            limited.extend(selected[: args.max_instances_per_partition])
        entries = limited

    records: list[dict[str, Any]] = []
    for seed in seeds:
        for entry in entries:
            instance = load_manifest_instance(dataset_root, entry)
            pool_best = references[instance.instance_id]
            records.append(
                evaluate_one(
                    instance,
                    pool_best=pool_best,
                    seed=seed,
                    num_samples=args.num_samples,
                    recovery_candidate_limit=args.recovery_candidate_limit,
                    recovery_max_moves=args.recovery_max_moves,
                )
            )

    records_path = output_root / "random_recovery_realistic_records.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    with (output_root / "random_recovery_realistic_records.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    by_partition = {
        partition: _summarize([record for record in records if record["partition"] == partition])
        for partition in partitions
    }
    summary = {
        "schema_version": "1.0",
        "method": "random_recovery",
        "dataset_root": str(dataset_root),
        "reference_records_root": str(_resolve(root, args.reference_records_root)),
        "output_root": str(output_root),
        "overall": _summarize(records),
        "by_partition": by_partition,
        "protocol": {
            "partitions": partitions,
            "seeds": seeds,
            "num_samples": int(args.num_samples),
            "recovery_candidate_limit": int(args.recovery_candidate_limit),
            "recovery_max_moves": int(args.recovery_max_moves),
            "policy": "verify random compatible proposals first; invoke recovery only if no raw proposal is verified; no constructive fallback",
        },
    }
    (output_root / "random_recovery_realistic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_root, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="artifacts/datasets/phase6e-e-realistic-profile")
    parser.add_argument(
        "--reference-records-root",
        default="artifacts/phase6e-e-realistic-profile-evaluation-10seed/records",
    )
    parser.add_argument("--output-root", default="artifacts/phase6f-random-recovery-realistic")
    parser.add_argument("--partitions", default=",".join(DEFAULT_PARTITIONS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--max-instances-per-partition", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--recovery-candidate-limit", type=int, default=16)
    parser.add_argument("--recovery-max-moves", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    overall = summary["overall"]
    print(
        "Random + Recovery: "
        f"records={overall['records']} "
        f"success={100.0 * overall['success_rate']:.2f}% "
        f"gap={100.0 * overall['mean_gap_to_pool_best']:.3f}% "
        f"time={overall['mean_total_seconds']:.3f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
