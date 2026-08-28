"""Cross-scale completion run for stronger deterministic heuristic baselines.

The Stage 3.9 cross-scale cloud run already evaluates Direct and Masked
Diffusion over ten seeds. This script complements that evidence with
deterministic non-learned baselines that do not depend on training seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.inference import (
    solve_latency_aware_heuristic,
    solve_local_search,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze


SCOPE = "phase6e_e_cross_scale_strong_heuristics"
DEFAULT_PARTITIONS = ("scale_medium", "scale_large", "scale_extra_large")
METHODS: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    (
        "latency_aware_heuristic",
        "Latency-aware heuristic",
        solve_latency_aware_heuristic,
    ),
    ("local_search", "Local search", solve_local_search),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _repo_root() / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("CSV argument cannot be empty.")
    return result


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


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def _finite_std(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return pstdev(finite) if len(finite) > 1 else (0.0 if finite else None)


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if row["success"]]
    return {
        "records": len(rows),
        "success_count": len(success),
        "success_rate": len(success) / len(rows) if rows else None,
        "mean_gap_to_pool_best": _finite_mean([row["gap_to_pool_best"] for row in rows]),
        "gap_std": _finite_std([row["gap_to_pool_best"] for row in rows]),
        "mean_total_seconds": _finite_mean([row["total_seconds"] for row in rows]),
        "source_rates": {
            source: (
                sum(1 for row in rows if row["source"] == source) / len(rows)
                if rows
                else None
            )
            for source in ("latency_aware_heuristic", "local_search", "failure")
        },
    }


def _paired(
    rows: list[Mapping[str, Any]],
    *,
    left: str,
    right: str,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    by_instance: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_instance[f"{row['partition']}:{row['instance_id']}"][row["method"]] = row
    left_wins = right_wins = ties = comparable = 0
    for methods in by_instance.values():
        lhs = methods.get(left)
        rhs = methods.get(right)
        if not lhs or not rhs or not lhs["success"] or not rhs["success"]:
            continue
        comparable += 1
        delta = float(lhs["objective"]) - float(rhs["objective"])
        if delta < -tolerance:
            left_wins += 1
        elif delta > tolerance:
            right_wins += 1
        else:
            ties += 1
    return {
        "left": left,
        "right": right,
        "comparable": comparable,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
    }


def _write_report(path: Path, evidence: Mapping[str, Any]) -> None:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * float(value):.3f}%"

    def ms(value: float | None) -> str:
        return "n/a" if value is None else f"{1000.0 * float(value):.2f}"

    lines = [
        "# Cross-Scale Strong Heuristic Completion",
        "",
        "This report complements the ten-seed Direct-vs-Masked cross-scale run",
        "with deterministic heuristic baselines that do not depend on training seeds.",
        "",
        "## Dataset",
        "",
        f"- Dataset root: `{evidence['protocol']['dataset_root']}`",
        f"- Partitions: `{', '.join(evidence['protocol']['partitions'])}`",
        f"- Instances: `{evidence['instances']}`",
        "",
        "## Partition Summary",
        "",
        "| Partition | Method | Success | Gap to pool best | Runtime (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    for partition, methods in evidence["by_partition"].items():
        for method_id, _label, _solver in METHODS:
            row = methods[method_id]
            lines.append(
                f"| {partition} | {method_id} | {pct(row['success_rate'])} | "
                f"{pct(row['mean_gap_to_pool_best'])} | {ms(row['mean_total_seconds'])} |"
            )
    pair = evidence["paired"]["local_vs_latency_aware"]
    lines.extend(
        [
            "",
            "## Paired Heuristic Check",
            "",
            "- Local search vs. latency-aware heuristic: "
            f"{pair['left_wins']}/{pair['ties']}/{pair['right_wins']} "
            f"over {pair['comparable']} comparable instances.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    dataset_root: Path,
    output_root: Path,
    partitions: tuple[str, ...],
    max_instances_per_partition: int | None,
) -> dict[str, Any]:
    freeze = audit_dataset_freeze(dataset_root)
    dataset_hash = _sha256(dataset_root / "dataset_freeze.json")
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=partitions,
        verify_checksum=True,
        require_freeze=True,
    )
    if max_instances_per_partition is None:
        indices = list(range(len(dataset)))
    else:
        per_partition: dict[str, int] = defaultdict(int)
        indices = []
        for index, item in enumerate(dataset):
            if per_partition[item.partition] < max_instances_per_partition:
                indices.append(index)
                per_partition[item.partition] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = perf_counter()
    for ordinal, index in enumerate(indices, 1):
        item = dataset[index]
        pool_best = float(np.min(item.pool.latencies))
        for method_id, _label, solver in METHODS:
            method_started = perf_counter()
            result = solver(item.instance)
            wall_seconds = perf_counter() - method_started
            objective = None if result.objective is None else float(result.objective)
            records.append(
                {
                    "scope": f"{SCOPE}_record",
                    "partition": item.partition,
                    "instance_id": item.instance.instance_id,
                    "num_services": item.instance.num_services,
                    "num_devices": item.instance.num_devices,
                    "method": method_id,
                    "success": bool(result.success),
                    "source": result.source if result.success else "failure",
                    "objective": objective,
                    "pool_best_objective": pool_best,
                    "gap_to_pool_best": None if objective is None else objective / pool_best - 1.0,
                    "total_seconds": float(result.metrics.get("total_seconds", wall_seconds)),
                    "wall_seconds": wall_seconds,
                    "metrics": _to_builtin(result.metrics),
                }
            )
        print(f"[{ordinal}/{len(indices)}] {item.partition}:{item.instance.instance_id}", flush=True)

    overall = {
        method_id: _aggregate([row for row in records if row["method"] == method_id])
        for method_id, _label, _solver in METHODS
    }
    by_partition = {}
    for partition in partitions:
        rows = [row for row in records if row["partition"] == partition]
        by_partition[partition] = {
            method_id: _aggregate([row for row in rows if row["method"] == method_id])
            for method_id, _label, _solver in METHODS
        }

    evidence = {
        "schema_version": "1.0",
        "scope": f"{SCOPE}_evidence",
        "protocol": {
            "dataset_root": str(dataset_root),
            "partitions": list(partitions),
            "max_instances_per_partition": max_instances_per_partition,
            "methods": {
                method_id: {"family": "deterministic_heuristic", "label": label}
                for method_id, label, _solver in METHODS
            },
        },
        "dataset_freeze_sha256": dataset_hash,
        "dataset_freeze": freeze,
        "instances": len(indices),
        "records": len(records),
        "total_wall_seconds": perf_counter() - started,
        "overall": overall,
        "by_partition": by_partition,
        "paired": {
            "local_vs_latency_aware": _paired(
                records,
                left="local_search",
                right="latency_aware_heuristic",
            )
        },
        "claim_boundary": (
            "Deterministic cross-scale heuristic completion. Combine with the "
            "separately frozen Direct-vs-Masked cross-scale evidence only after "
            "checking dataset/protocol compatibility."
        ),
    }
    write_json(output_root / "cross_scale_strong_heuristics_records.json", {"records": records})
    write_json(output_root / "cross_scale_strong_heuristics_evidence.json", evidence)
    _write_report(output_root / "cross_scale_strong_heuristics_report.md", evidence)
    return {
        "evidence": str(output_root / "cross_scale_strong_heuristics_evidence.json"),
        "records": len(records),
        "instances": len(indices),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="artifacts/datasets/phase6c-final-scale",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase6e-e-cross-scale-strong-heuristics",
    )
    parser.add_argument("--partitions", default=",".join(DEFAULT_PARTITIONS))
    parser.add_argument("--max-instances-per-partition", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_instances_per_partition is not None and args.max_instances_per_partition < 1:
        raise ValueError("--max-instances-per-partition must be positive.")
    print(
        run(
            dataset_root=_resolve(args.dataset_root),
            output_root=_resolve(args.output_root),
            partitions=_parse_csv(args.partitions, DEFAULT_PARTITIONS),
            max_instances_per_partition=args.max_instances_per_partition,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
