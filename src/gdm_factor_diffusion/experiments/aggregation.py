"""Aggregate raw experiment records without discarding per-instance evidence."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from gdm_factor_diffusion.common.logging import write_json

from .schema import stable_fingerprint

SUMMARY_METRICS = (
    "gap_to_pool_best",
    "objective",
    "success",
    "output_verified",
    "raw_feasible_rate",
    "raw_feasible_count",
    "raw_any_feasible",
    "raw_unique_count",
    "raw_unique_rate",
    "raw_pairwise_hamming",
    "best_raw_objective",
    "best_raw_gap_to_pool_best",
    "raw_capacity_violation_rate",
    "raw_link_violation_rate",
    "repair_attempts",
    "repair_successes",
    "repair_success_rate",
    "total_repair_moves",
    "pre_fallback_success",
    "best_pre_fallback_objective",
    "best_pre_fallback_gap_to_pool_best",
    "fallback_invoked",
    "fallback_success",
    "fallback_search_nodes",
    "milp_gap",
    "milp_optimal",
    "milp_solver_runtime_seconds",
    "time_limit_exceeded",
    "sampling_seconds",
    "optimization_seconds",
    "verification_seconds",
    "repair_seconds",
    "fallback_seconds",
    "exact_evaluation_seconds",
    "selection_seconds",
    "total_seconds",
)


def _values(records: Iterable[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(metric)
        if value is None:
            value = record.get("metrics", {}).get(metric)
        if value is not None:
            values.append(float(value))
    return values


def _descriptive(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    deviation = stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean(values),
        "std": deviation,
        "minimum": min(values),
        "maximum": max(values),
        "ci95_half_width": (
            1.96 * deviation / math.sqrt(len(values)) if len(values) > 1 else 0.0
        ),
    }


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "instances": len(records),
        "successes": sum(bool(record["success"]) for record in records),
        "failures": sum(not bool(record["success"]) for record in records),
        "selection_sources": dict(sorted(Counter(r["source"] for r in records).items())),
        "metrics": {
            metric: _descriptive(_values(records, metric))
            for metric in SUMMARY_METRICS
        },
    }


def pairwise_outcomes(
    records: list[dict[str, Any]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, dict[str, Any]]:
    """Compare every method pair without discarding method failures."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    by_instance: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        method_id = str(record["method_id"])
        instance_id = str(record["instance_id"])
        if method_id in by_instance[instance_id]:
            raise ValueError("Pairwise comparison requires one record per method/instance.")
        by_instance[instance_id][method_id] = record
    methods = sorted({str(record["method_id"]) for record in records})
    output: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            counts = {
                "instances": 0,
                "both_success": 0,
                "left_wins": 0,
                "ties": 0,
                "right_wins": 0,
                "left_only_success": 0,
                "right_only_success": 0,
                "both_failed": 0,
            }
            for methods_for_instance in by_instance.values():
                if left not in methods_for_instance or right not in methods_for_instance:
                    continue
                counts["instances"] += 1
                left_record = methods_for_instance[left]
                right_record = methods_for_instance[right]
                left_success = bool(left_record["success"])
                right_success = bool(right_record["success"])
                if left_success and right_success:
                    counts["both_success"] += 1
                    left_objective = float(left_record["objective"])
                    right_objective = float(right_record["objective"])
                    if left_objective < right_objective - tolerance:
                        counts["left_wins"] += 1
                    elif right_objective < left_objective - tolerance:
                        counts["right_wins"] += 1
                    else:
                        counts["ties"] += 1
                elif left_success:
                    counts["left_only_success"] += 1
                elif right_success:
                    counts["right_only_success"] += 1
                else:
                    counts["both_failed"] += 1
            output[f"{left}__vs__{right}"] = {
                "left_method": left,
                "right_method": right,
                **counts,
            }
    return output


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty experiment record set.")
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_partition: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_method[record["method_id"]].append(record)
        by_partition[record["partition"]][record["method_id"]].append(record)
    return {
        "quality_conditioning": "objective_and_gap_use_successful_outputs_only",
        "methods": {
            method: _group_summary(values)
            for method, values in sorted(by_method.items())
        },
        "partitions": {
            partition: {
                method: _group_summary(values)
                for method, values in sorted(methods.items())
            }
            for partition, methods in sorted(by_partition.items())
        },
        "pairwise": pairwise_outcomes(records),
        "pairwise_by_partition": {
            partition: pairwise_outcomes(
                [record for record in records if record["partition"] == partition]
            )
            for partition in sorted(by_partition)
        },
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quality_fingerprint(records: list[dict[str, Any]]) -> str:
    quality_fields = [
        {
            "instance_id": record["instance_id"],
            "method_id": record["method_id"],
            "method_seed": record["method_seed"],
            "success": record["success"],
            "source": record["source"],
            "objective": record["objective"],
            "gap_to_pool_best": record["gap_to_pool_best"],
            "raw_feasible_count": record["metrics"]["raw_feasible_count"],
            "repair_successes": record["metrics"]["repair_successes"],
            "fallback_success": record["metrics"]["fallback_success"],
        }
        for record in records
    ]
    return stable_fingerprint(quality_fields)


def write_record_csv(path: str | Path, records: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    explicit_fields = (
        "instance_id",
        "partition",
        "regime",
        "method_id",
        "method_kind",
        "method_seed",
        "success",
        "output_verified",
        "source",
        "objective",
        "pool_best",
        "gap_to_pool_best",
        "num_services",
        "num_devices",
        "num_dependencies",
        "candidate_edges",
        "time_limit_seconds",
        "time_limit_scope",
        "time_limit_observed_seconds",
        "time_limit_exceeded",
    )
    metric_fields = tuple(
        metric for metric in SUMMARY_METRICS if metric not in explicit_fields
    )
    fields = explicit_fields + metric_fields
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields}
            for key in metric_fields:
                row[key] = record["metrics"].get(key)
            writer.writerow(row)
    return destination


def write_aggregate_csv(path: str | Path, aggregate: dict[str, Any]) -> Path:
    """Write one flat paper-table source row per scope and method."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scope",
        "partition",
        "method_id",
        "instances",
        "successes",
        "failures",
    ) + tuple(
        f"{metric}_{statistic}"
        for metric in SUMMARY_METRICS
        for statistic in ("mean", "std", "ci95_half_width")
    )
    rows: list[dict[str, Any]] = []

    def add_row(
        scope: str,
        partition: str | None,
        method_id: str,
        summary: dict[str, Any],
    ) -> None:
        row: dict[str, Any] = {
            "scope": scope,
            "partition": partition,
            "method_id": method_id,
            "instances": summary["instances"],
            "successes": summary["successes"],
            "failures": summary["failures"],
        }
        for metric, values in summary["metrics"].items():
            if values is None:
                continue
            for statistic in ("mean", "std", "ci95_half_width"):
                row[f"{metric}_{statistic}"] = values[statistic]
        rows.append(row)

    for method_id, summary in aggregate["methods"].items():
        add_row("all", None, method_id, summary)
    for partition, methods in aggregate["partitions"].items():
        for method_id, summary in methods.items():
            add_row("partition", partition, method_id, summary)

    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def aggregate_run_directories(
    run_directories: Iterable[str | Path],
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    runs = [Path(path) for path in run_directories]
    if len(runs) < 2:
        raise ValueError("At least two experiment runs are required.")
    summaries = []
    resolved_manifests = []
    coverage: set[tuple[str, str]] | None = None
    instance_ids: set[str] | None = None
    for run in runs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        resolved = json.loads(
            (run / "resolved_manifest.json").read_text(encoding="utf-8")
        )
        records = read_jsonl(run / "records.jsonl")
        current = {(r["instance_id"], r["method_id"]) for r in records}
        current_instances = {r["instance_id"] for r in records}
        if coverage is None:
            coverage = current
            instance_ids = current_instances
        elif current != coverage:
            raise ValueError("Experiment runs do not cover identical methods/instances.")
        summaries.append(summary)
        resolved_manifests.append(resolved)

    reference = resolved_manifests[0]
    reference_methods = [
        {
            key: value
            for key, value in method.items()
            if key not in {"checkpoint"}
        }
        for method in reference["manifest"]["methods"]
    ]
    for resolved in resolved_manifests[1:]:
        methods = [
            {
                key: value
                for key, value in method.items()
                if key not in {"checkpoint"}
            }
            for method in resolved["manifest"]["methods"]
        ]
        if resolved["dataset_freeze_sha256"] != reference["dataset_freeze_sha256"]:
            raise ValueError("Experiment runs use different frozen datasets.")
        if resolved["instance_order"] != reference["instance_order"]:
            raise ValueError("Experiment runs use different instance orders.")
        if resolved["device_resolved"] != reference["device_resolved"]:
            raise ValueError("Experiment runs use different evaluation devices.")
        if methods != reference_methods:
            raise ValueError("Experiment runs use different method budgets.")

    method_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pairwise_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    partition_metrics: dict[
        str, dict[str, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    partition_pairwise_metrics: dict[
        str, dict[str, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for summary in summaries:
        for method, method_summary in summary["aggregate"]["methods"].items():
            for metric, values in method_summary["metrics"].items():
                if values is not None:
                    method_metrics[method][metric].append(float(values["mean"]))
        for pair, values in summary["aggregate"]["pairwise"].items():
            for metric in (
                "both_success",
                "left_wins",
                "ties",
                "right_wins",
                "left_only_success",
                "right_only_success",
                "both_failed",
            ):
                pairwise_metrics[pair][metric].append(float(values[metric]))
        for partition, methods in summary["aggregate"]["partitions"].items():
            for method, method_summary in methods.items():
                for metric, values in method_summary["metrics"].items():
                    if values is not None:
                        partition_metrics[partition][method][metric].append(
                            float(values["mean"])
                        )
        for partition, pairs in summary["aggregate"]["pairwise_by_partition"].items():
            for pair, values in pairs.items():
                for metric in (
                    "both_success",
                    "left_wins",
                    "ties",
                    "right_wins",
                    "left_only_success",
                    "right_only_success",
                    "both_failed",
                ):
                    partition_pairwise_metrics[partition][pair][metric].append(
                        float(values[metric])
                    )
    seeds = [int(resolved["manifest"]["seed"]) for resolved in resolved_manifests]
    payload = {
        "schema_version": "1.0",
        "run_directories": [str(path.resolve()) for path in runs],
        "runs": len(runs),
        "unique_seeds": len(set(seeds)),
        "seed_values": seeds,
        "aggregation_scope": (
            "multi_seed" if len(set(seeds)) == len(seeds) else "repeated_runs"
        ),
        "instances_per_run": len(instance_ids or ()),
        "records_per_run": len(coverage or ()),
        "dataset_freeze_sha256": reference["dataset_freeze_sha256"],
        "device_resolved": reference["device_resolved"],
        "methods": {
            method: {
                metric: _descriptive(values)
                for metric, values in sorted(metrics.items())
            }
            for method, metrics in sorted(method_metrics.items())
        },
        "pairwise": {
            pair: {
                metric: _descriptive(values)
                for metric, values in sorted(metrics.items())
            }
            for pair, metrics in sorted(pairwise_metrics.items())
        },
        "partitions": {
            partition: {
                method: {
                    metric: _descriptive(values)
                    for metric, values in sorted(metrics.items())
                }
                for method, metrics in sorted(methods.items())
            }
            for partition, methods in sorted(partition_metrics.items())
        },
        "pairwise_by_partition": {
            partition: {
                pair: {
                    metric: _descriptive(values)
                    for metric, values in sorted(metrics.items())
                }
                for pair, metrics in sorted(pairs.items())
            }
            for partition, pairs in sorted(partition_pairwise_metrics.items())
        },
        "quality_conditioning": "objective_and_gap_use_successful_outputs_only",
    }
    if output is not None:
        destination = write_json(output, payload)
        csv_path = destination.with_suffix(".csv")
        fields = (
            "scope",
            "partition",
            "method_id",
            "metric",
            "count",
            "mean",
            "std",
            "minimum",
            "maximum",
            "ci95_half_width",
        )
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for method_id, metrics in payload["methods"].items():
                for metric, values in metrics.items():
                    if values is None:
                        continue
                    writer.writerow(
                        {
                            "scope": "all",
                            "partition": None,
                            "method_id": method_id,
                            "metric": metric,
                            **values,
                        }
                    )
            for partition, methods in payload["partitions"].items():
                for method_id, metrics in methods.items():
                    for metric, values in metrics.items():
                        if values is None:
                            continue
                        writer.writerow(
                            {
                                "scope": "partition",
                                "partition": partition,
                                "method_id": method_id,
                                "metric": metric,
                                **values,
                            }
                        )
    return payload
