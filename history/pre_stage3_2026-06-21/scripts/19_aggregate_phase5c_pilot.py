"""Aggregate multi-seed pilot evaluations and emit the Phase 5C decision data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from gdm_factor_diffusion.common.logging import write_json


METHODS = ("learned_hybrid", "random_hybrid", "fallback_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _scope_summary(
    run_records: list[list[dict[str, Any]]],
    *,
    include_validation: bool,
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    expected_instances: set[str] | None = None
    for records in run_records:
        selected = [
            record
            for record in records
            if include_validation or record["partition"] != "validation"
        ]
        by_method = {
            method: {
                record["instance_id"]: record
                for record in selected
                if record["method"] == method
            }
            for method in METHODS
        }
        instance_ids = set(by_method["learned_hybrid"])
        if any(set(by_method[method]) != instance_ids for method in METHODS):
            raise ValueError("Evaluation methods do not cover identical instances.")
        if expected_instances is None:
            expected_instances = instance_ids
        elif instance_ids != expected_instances:
            raise ValueError("Evaluation seeds do not cover identical instances.")

        wins = ties = losses = 0
        for instance_id in instance_ids:
            learned = by_method["learned_hybrid"][instance_id]["objective"]
            fallback = by_method["fallback_only"][instance_id]["objective"]
            difference = learned - fallback
            if difference < -1e-12:
                wins += 1
            elif abs(difference) <= 1e-12:
                ties += 1
            else:
                losses += 1

        method_metrics: dict[str, dict[str, float]] = {}
        for method, by_instance in by_method.items():
            values = list(by_instance.values())
            method_metrics[method] = {
                "mean_gap_to_pool_best": mean(
                    float(record["gap_to_pool_best"]) for record in values
                ),
                "mean_total_seconds": mean(
                    float(record["metrics"]["total_seconds"]) for record in values
                ),
                "final_success_rate": mean(
                    float(record["success"]) for record in values
                ),
            }
            raw_rates = [
                record["metrics"]["raw_feasible_rate"]
                for record in values
                if record["metrics"]["raw_feasible_rate"] is not None
            ]
            method_metrics[method]["mean_raw_feasible_rate"] = (
                mean(float(value) for value in raw_rates) if raw_rates else 0.0
            )
        per_seed.append(
            {
                "instances": len(instance_ids),
                "learned_vs_fallback": {
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                },
                "methods": method_metrics,
            }
        )

    aggregate: dict[str, Any] = {
        "seeds": len(per_seed),
        "instances_per_seed": per_seed[0]["instances"],
        "learned_vs_fallback": {
            outcome: _mean_std(
                [
                    float(seed["learned_vs_fallback"][outcome])
                    for seed in per_seed
                ]
            )
            for outcome in ("wins", "ties", "losses")
        },
        "methods": {},
    }
    for method in METHODS:
        aggregate["methods"][method] = {
            metric: _mean_std(
                [float(seed["methods"][method][metric]) for seed in per_seed]
            )
            for metric in (
                "mean_gap_to_pool_best",
                "mean_total_seconds",
                "final_success_rate",
                "mean_raw_feasible_rate",
            )
        }
    fallback_gap = aggregate["methods"]["fallback_only"]["mean_gap_to_pool_best"]["mean"]
    learned_gap = aggregate["methods"]["learned_hybrid"]["mean_gap_to_pool_best"]["mean"]
    aggregate["relative_gap_reduction_vs_fallback"] = (
        (fallback_gap - learned_gap) / fallback_gap if fallback_gap else 0.0
    )
    aggregate["per_seed"] = per_seed
    return aggregate


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.resolve()
    run_directories = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "summary.json").exists()
    )
    if len(run_directories) < 2:
        raise ValueError("At least two completed evaluation runs are required.")

    checkpoints: list[str] = []
    run_records: list[list[dict[str, Any]]] = []
    for run_directory in run_directories:
        summary = _read_json(run_directory / "summary.json")
        checkpoints.append(summary["checkpoint"])
        run_records.append(_read_jsonl(run_directory / "records.jsonl"))
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Evaluation runs must use distinct checkpoints.")

    test_only = _scope_summary(run_records, include_validation=False)
    all_holdout = _scope_summary(run_records, include_validation=True)
    decision = {
        "go": (
            test_only["methods"]["learned_hybrid"]["final_success_rate"]["minimum"] == 1.0
            and test_only["learned_vs_fallback"]["losses"]["maximum"] == 0.0
            and test_only["relative_gap_reduction_vs_fallback"] > 0.0
        ),
        "basis": (
            "GO requires every pilot seed to preserve final feasibility, never "
            "lose to always-available fallback, and reduce mean test-only gap."
        ),
    }
    payload = {
        "schema_version": "1.0",
        "evaluation_runs": [str(path) for path in run_directories],
        "checkpoints": checkpoints,
        "test_only": test_only,
        "including_validation": all_holdout,
        "decision": decision,
    }
    destination = args.output or root / "phase5c_pilot_aggregate.json"
    write_json(destination, payload)
    print(
        f"runs={len(run_directories)} test_instances={test_only['instances_per_seed']} "
        f"learned_gap={test_only['methods']['learned_hybrid']['mean_gap_to_pool_best']['mean']:.6f} "
        f"fallback_gap={test_only['methods']['fallback_only']['mean_gap_to_pool_best']['mean']:.6f} "
        f"gap_reduction={test_only['relative_gap_reduction_vs_fallback']:.2%} "
        f"decision={'GO' if decision['go'] else 'NO-GO'}"
    )


if __name__ == "__main__":
    main()
