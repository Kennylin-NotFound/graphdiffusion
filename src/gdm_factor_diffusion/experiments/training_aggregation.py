"""Audit, aggregate, and freeze independent production-training seeds."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import torch

from gdm_factor_diffusion.common.logging import write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _descriptive(values: list[float]) -> dict[str, float]:
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


def _training_signature(config: dict[str, Any]) -> dict[str, Any]:
    signature = deepcopy(config)
    training = signature["training"]
    training.pop("seed", None)
    training.pop("run_name", None)
    return signature


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def audit_training_run(
    run_directory: str | Path,
    *,
    expected_steps: int,
) -> dict[str, Any]:
    """Return a strict evidence record for one completed independent seed."""

    run = Path(run_directory).resolve()
    summary = _read_json(run / "summary.json")
    if summary["termination_reason"] != "completed":
        raise ValueError(f"Training run is not complete: {run}")
    if int(summary["completed_steps"]) != expected_steps:
        raise ValueError(f"Training run has an unexpected completed step: {run}")
    best_path = run / "best_checkpoint.pt"
    payload = _checkpoint_payload(best_path)
    metadata = payload["metadata"]
    config = metadata["config"]
    model_kind = str(config["training"].get("model_kind", "diffusion"))
    seed = int(config["training"]["seed"])
    best_step = int(summary["selection_state"]["best_step"])
    if int(payload["step"]) != best_step:
        raise ValueError(f"Best checkpoint and summary step disagree: {run}")
    if int(config["training"]["steps"]) != expected_steps:
        raise ValueError(f"Best checkpoint uses an unexpected step budget: {run}")
    constrained = summary["selection_state"]["best_constrained_metrics"]
    if int(constrained["step"]) != best_step:
        raise ValueError(f"Best constrained metric and checkpoint step disagree: {run}")
    if float(constrained["verified_rate"]) != 1.0:
        raise ValueError(f"Best checkpoint does not preserve final feasibility: {run}")
    metrics = _read_jsonl(run / "metrics.jsonl")
    return {
        "run_directory": str(run),
        "seed": seed,
        "model_kind": model_kind,
        "completed_steps": expected_steps,
        "best_step": best_step,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": _sha256(best_path),
        "metrics_sha256": _sha256(run / "metrics.jsonl"),
        "summary_sha256": _sha256(run / "summary.json"),
        "dataset_freeze_sha256": config["training"]["resolved_dataset_freeze_sha256"],
        "dataset_core_sha256": config["training"]["resolved_dataset_core_sha256"],
        "training_signature": _training_signature(config),
        "best_metrics": {
            "verified_rate": float(constrained["verified_rate"]),
            "mean_gap_to_pool_best": float(constrained["mean_gap_to_pool_best"]),
            "mean_raw_feasible_rate": float(constrained["mean_raw_feasible_rate"]),
            "learned_wins_over_fallback": int(
                constrained["learned_wins_over_fallback"]
            ),
            "learned_ties_with_fallback": int(
                constrained["learned_ties_with_fallback"]
            ),
            "instances": int(constrained["instances"]),
            "clean_state_validation_loss": float(
                summary["selection_state"]["best_denoising_loss"]
            ),
        },
        "metrics": metrics,
    }


def _aggregate_curves(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = {
        "validation_denoising": ("loss_total", "clean_accuracy"),
        "validation_constrained": (
            "mean_gap_to_pool_best",
            "mean_raw_feasible_rate",
            "learned_wins_over_fallback",
        ),
    }
    long_records: list[dict[str, Any]] = []
    curve_summary: list[dict[str, Any]] = []
    for split, names in metrics.items():
        per_seed: dict[int, dict[int, dict[str, Any]]] = {}
        for run in runs:
            selected = {
                int(record["step"]): record
                for record in run["metrics"]
                if record["split"] == split
            }
            per_seed[run["seed"]] = selected
        common_steps = set.intersection(
            *(set(records) for records in per_seed.values())
        )
        if not common_steps:
            raise ValueError(f"Training runs share no common steps for {split}.")
        for step in sorted(common_steps):
            for metric in names:
                values = [
                    float(per_seed[seed][step][metric]) for seed in sorted(per_seed)
                ]
                descriptive = _descriptive(values)
                row = {
                    "split": split,
                    "step": step,
                    "metric": metric,
                    **descriptive,
                }
                curve_summary.append(row)
                for seed, value in zip(sorted(per_seed), values, strict=True):
                    long_records.append(
                        {
                            "split": split,
                            "step": step,
                            "metric": metric,
                            "seed": seed,
                            "value": value,
                        }
                    )
    return curve_summary, long_records


def _write_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    records = list(records)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _export_figure(output: Path, curves: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8})
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(8.2, 2.5),
        constrained_layout=True,
    )
    specs = (
        ("validation_denoising", "loss_total", "Validation loss", 1.0),
        (
            "validation_constrained",
            "mean_gap_to_pool_best",
            "Verified gap (%)",
            100.0,
        ),
        (
            "validation_constrained",
            "mean_raw_feasible_rate",
            "Raw feasible rate (%)",
            100.0,
        ),
    )
    for axis, (split, metric, label, scale) in zip(axes, specs, strict=True):
        rows = sorted(
            (
                row
                for row in curves
                if row["split"] == split and row["metric"] == metric
            ),
            key=lambda row: row["step"],
        )
        x = [row["step"] for row in rows]
        y = [scale * row["mean"] for row in rows]
        deviation = [scale * row["std"] for row in rows]
        axis.plot(x, y)
        axis.fill_between(
            x,
            [value - spread for value, spread in zip(y, deviation, strict=True)],
            [value + spread for value, spread in zip(y, deviation, strict=True)],
            alpha=0.2,
        )
        axis.set_xlabel("Training step")
        axis.set_ylabel(label)
        axis.grid(True, linestyle=":", linewidth=0.5)
    paths = []
    for suffix in (".png", ".pdf"):
        path = output / f"five_seed_training_curves{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def verify_checkpoint_freeze(path: str | Path) -> dict[str, Any]:
    """Verify every checkpoint and run-evidence hash in a freeze manifest."""

    freeze_path = Path(path)
    freeze = _read_json(freeze_path)
    if freeze.get("scope") not in {
        "five_independent_training_seeds",
        "independent_training_seeds",
    }:
        raise ValueError("Unsupported checkpoint-freeze scope.")
    seeds: set[int] = set()
    for run in freeze["runs"]:
        seed = int(run["seed"])
        if seed in seeds:
            raise ValueError("Checkpoint freeze contains duplicate seeds.")
        seeds.add(seed)
        for key, hash_key in (
            ("best_checkpoint", "best_checkpoint_sha256"),
            ("metrics", "metrics_sha256"),
            ("summary", "summary_sha256"),
        ):
            path_value = (
                Path(run["best_checkpoint"])
                if key == "best_checkpoint"
                else Path(run["run_directory"]) / f"{key}.jsonl"
                if key == "metrics"
                else Path(run["run_directory"]) / "summary.json"
            )
            if _sha256(path_value) != run[hash_key]:
                raise ValueError(f"Checkpoint-freeze hash mismatch: {path_value}")
    if seeds != set(int(seed) for seed in freeze["seeds"]):
        raise ValueError("Checkpoint-freeze seed list is inconsistent.")
    return freeze


def aggregate_and_freeze_training_runs(
    run_directories: Iterable[str | Path],
    *,
    expected_seeds: Iterable[int],
    expected_steps: int,
    output_directory: str | Path,
    scope: str = "five_independent_training_seeds",
) -> dict[str, Any]:
    """Aggregate independent seeds and write a cryptographic checkpoint freeze."""

    if scope not in {
        "five_independent_training_seeds",
        "independent_training_seeds",
    }:
        raise ValueError("Unsupported checkpoint-freeze scope.")
    expected_seed_values = tuple(int(seed) for seed in expected_seeds)
    runs = [
        audit_training_run(run, expected_steps=expected_steps)
        for run in run_directories
    ]
    seeds = tuple(run["seed"] for run in runs)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Training runs must use distinct seeds.")
    if set(seeds) != set(expected_seed_values):
        raise ValueError("Training runs do not match the expected seed contract.")
    signatures = {
        json.dumps(run["training_signature"], sort_keys=True) for run in runs
    }
    if len(signatures) != 1:
        raise ValueError("Training runs disagree beyond seed and run_name.")
    if len({run["dataset_freeze_sha256"] for run in runs}) != 1:
        raise ValueError("Training runs use different frozen datasets.")
    curves, long_records = _aggregate_curves(runs)
    best_summary = {
        "best_step": _descriptive([float(run["best_step"]) for run in runs]),
        **{
            metric: _descriptive(
                [float(run["best_metrics"][metric]) for run in runs]
            )
            for metric in (
                "mean_gap_to_pool_best",
                "mean_raw_feasible_rate",
                "learned_wins_over_fallback",
                "clean_state_validation_loss",
            )
        },
    }
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = {
        "schema_version": "1.0",
        "scope": scope,
        "expected_steps": expected_steps,
        "seeds": sorted(seeds),
        "dataset_freeze_sha256": runs[0]["dataset_freeze_sha256"],
        "dataset_core_sha256": runs[0]["dataset_core_sha256"],
        "runs": [
            {key: value for key, value in run.items() if key not in {"metrics", "training_signature"}}
            for run in sorted(runs, key=lambda item: item["seed"])
        ],
        "best_checkpoint_summary": best_summary,
    }
    write_json(output / "checkpoint_freeze.json", freeze)
    write_json(output / "training_curve_summary.json", curves)
    _write_csv(output / "training_curve_summary.csv", curves)
    _write_csv(output / "training_curve_long.csv", long_records)
    figure_paths = _export_figure(output, curves)
    payload = {
        **freeze,
        "training_curve_records": len(curves),
        "figure_paths": figure_paths,
    }
    write_json(output / "five_seed_summary.json", payload)
    return payload
