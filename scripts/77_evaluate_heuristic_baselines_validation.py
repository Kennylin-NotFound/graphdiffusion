"""Evaluate deterministic non-learned baselines on the validation split.

This script is intentionally independent from the learned Stage 3.9 campaigns.
It provides manuscript-facing evidence for traditional baselines that can be
rerun without checkpoints or GPU state.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Callable

import numpy as np

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.inference import (
    solve_fallback_only,
    solve_greedy_local,
    solve_latency_aware_heuristic,
    solve_local_search,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze


METHODS: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ("fallback_only", "Fallback only", solve_fallback_only),
    ("greedy_local", "Greedy", solve_greedy_local),
    (
        "latency_aware_heuristic",
        "Latency-aware heuristic",
        solve_latency_aware_heuristic,
    ),
    ("local_search", "Local search", solve_local_search),
)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summarize_method(records: list[dict[str, Any]]) -> dict[str, Any]:
    success = [record for record in records if record["success"]]
    gaps = [float(record["gap_to_pool_best"]) for record in success]
    objectives = [float(record["objective"]) for record in success]
    runtimes = [float(record["total_seconds"]) for record in records]
    sources = Counter(str(record["source"]) for record in records)
    return {
        "count": len(records),
        "success_count": len(success),
        "success_rate": len(success) / len(records) if records else None,
        "mean_objective": mean(objectives) if objectives else None,
        "mean_gap_to_pool_best": mean(gaps) if gaps else None,
        "std_gap_to_pool_best": pstdev(gaps) if len(gaps) > 1 else 0.0,
        "mean_gap_percent": 100.0 * mean(gaps) if gaps else None,
        "std_gap_percent": 100.0 * pstdev(gaps) if len(gaps) > 1 else 0.0,
        "mean_total_seconds": mean(runtimes) if runtimes else None,
        "source_counts": dict(sorted(sources.items())),
    }


def _paired_comparison(
    by_instance: dict[str, dict[str, dict[str, Any]]],
    *,
    left: str,
    right: str,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    left_better = 0
    right_better = 0
    tie = 0
    comparable = 0
    for row in by_instance.values():
        lhs = row.get(left)
        rhs = row.get(right)
        if not lhs or not rhs or not lhs["success"] or not rhs["success"]:
            continue
        comparable += 1
        delta = float(lhs["objective"]) - float(rhs["objective"])
        if delta < -tolerance:
            left_better += 1
        elif delta > tolerance:
            right_better += 1
        else:
            tie += 1
    return {
        "left": left,
        "right": right,
        "comparable": comparable,
        "left_better": left_better,
        "right_better": right_better,
        "tie": tie,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Heuristic Baseline Validation",
        "",
        f"- Dataset: `{summary['dataset_root']}`",
        f"- Partition: `{summary['partition']}`",
        f"- Instances: {summary['num_instances']}",
        f"- Output root: `{summary['output_root']}`",
        "",
        "| Method | Success | Gap to pool-best (%) | Runtime (ms) | Source counts |",
        "|---|---:|---:|---:|---|",
    ]
    for method_id, label, _ in METHODS:
        row = summary["methods"][method_id]
        success = f"{row['success_count']}/{row['count']} ({100.0 * row['success_rate']:.1f}%)"
        gap = (
            "n/a"
            if row["mean_gap_percent"] is None
            else f"{row['mean_gap_percent']:.3f} +/- {row['std_gap_percent']:.3f}"
        )
        runtime = (
            "n/a"
            if row["mean_total_seconds"] is None
            else f"{1000.0 * row['mean_total_seconds']:.2f}"
        )
        sources = ", ".join(
            f"{name}: {count}" for name, count in row["source_counts"].items()
        )
        lines.append(f"| {label} | {success} | {gap} | {runtime} | {sources} |")

    paired = summary["paired"]
    lines.extend(
        [
            "",
            "## Paired Checks",
            "",
            "- Local search vs. latency-aware heuristic: "
            f"local better/tie/heuristic better = "
            f"{paired['local_vs_heuristic']['left_better']}/"
            f"{paired['local_vs_heuristic']['tie']}/"
            f"{paired['local_vs_heuristic']['right_better']} over "
            f"{paired['local_vs_heuristic']['comparable']} comparable instances.",
            "",
            "## Notes",
            "",
            "- `latency_aware_heuristic` is a deterministic constructive baseline; "
            "it is not a learned model and uses only instance-level latency, "
            "capacity, compatibility, and dependency information.",
            "- `local_search` starts from the best feasible constructive/fallback "
            "candidate and accepts only strict one-service relocation improvements.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("implementation/artifacts/datasets/phase6e-e-stage3-development"),
    )
    parser.add_argument("--partition", default="checkpoint_selection")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("implementation/artifacts/phase6e-e-heuristic-baselines-validation"),
    )
    parser.add_argument("--instance-limit", type=int, default=None)
    parser.add_argument("--skip-freeze-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_freeze_audit:
        audit_dataset_freeze(root)

    dataset = LabeledDeploymentDataset(
        root,
        partitions=[args.partition],
        verify_checksum=True,
        require_freeze=False,
    )
    limit = len(dataset) if args.instance_limit is None else min(args.instance_limit, len(dataset))
    records: list[dict[str, Any]] = []
    by_instance: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    started = perf_counter()

    for index in range(limit):
        item = dataset[index]
        pool_best = float(item.pool.latencies[0])
        for method_id, _, solver in METHODS:
            method_start = perf_counter()
            result = solver(item.instance)
            elapsed = perf_counter() - method_start
            objective = None if result.objective is None else float(result.objective)
            record = {
                "instance_index": index,
                "instance_id": item.instance.instance_id,
                "partition": item.partition,
                "num_services": item.instance.num_services,
                "num_devices": item.instance.num_devices,
                "pool_best_objective": pool_best,
                "method": method_id,
                "success": result.success,
                "source": result.source,
                "objective": objective,
                "gap_to_pool_best": (
                    None if objective is None else objective / pool_best - 1.0
                ),
                "total_seconds": float(result.metrics.get("total_seconds", elapsed)),
                "wall_seconds": elapsed,
                "metrics": _to_builtin(result.metrics),
            }
            records.append(record)
            by_instance[item.instance.instance_id][method_id] = record
        print(f"[{index + 1}/{limit}] {item.instance.instance_id}", flush=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)

    summary = {
        "dataset_root": str(root),
        "partition": args.partition,
        "num_instances": limit,
        "output_root": str(output_root),
        "total_wall_seconds": perf_counter() - started,
        "methods": {
            method_id: _summarize_method(grouped[method_id])
            for method_id, _, _ in METHODS
        },
        "paired": {
            "local_vs_heuristic": _paired_comparison(
                by_instance,
                left="local_search",
                right="latency_aware_heuristic",
            )
        },
    }
    write_json(output_root / "records.json", {"records": records})
    write_json(output_root / "summary.json", summary)
    _write_report(output_root / "report_zh.md", summary)
    print(json.dumps(summary["methods"], indent=2), flush=True)


if __name__ == "__main__":
    main()
