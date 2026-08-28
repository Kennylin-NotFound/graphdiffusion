"""Paper-ready diagnostic figures generated only from raw experiment artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .aggregation import read_jsonl

TIMING_STACK = (
    ("sampling_seconds", "Sampling"),
    ("optimization_seconds", "Optimization"),
    ("verification_seconds", "Verification"),
    ("repair_seconds", "Repair"),
    ("fallback_seconds", "Fallback"),
    ("exact_evaluation_seconds", "Exact evaluation"),
    ("selection_seconds", "Selection"),
)


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required for Phase 6 paper-ready figure export."
        ) from error
    return plt


def _save(figure, output: Path, stem: str) -> list[str]:
    paths = []
    for suffix in (".png", ".pdf"):
        path = output / f"{stem}{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(str(path.resolve()))
    return paths


def _group(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    return dict(sorted(grouped.items()))


def _display_name(identifier: str) -> str:
    return identifier.replace("_", " ")


def _available_values(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def export_run_figures(
    run_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Export a compact standard figure set for one completed evaluation run."""

    plt = _matplotlib()
    run = Path(run_directory)
    records = read_jsonl(run / "records.jsonl")
    if not records:
        raise ValueError("The experiment run contains no records.")
    output = Path(output_directory) if output_directory is not None else run / "figures"
    output.mkdir(parents=True, exist_ok=True)
    by_method = _group(records, "method_id")
    method_ids = list(by_method)
    generated: dict[str, list[str]] = {}

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "figure.figsize": (3.5, 2.5),
        }
    )

    figure, axis = plt.subplots()
    for method, values in by_method.items():
        gaps = _available_values(values, "gap_to_pool_best")
        if not gaps:
            continue
        axis.scatter(
            mean(float(record["metrics"]["total_seconds"]) for record in values),
            100.0 * mean(gaps),
            label=_display_name(method),
            s=28,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Mean online solving time (s)")
    axis.set_ylabel("Mean gap to pool best (%)")
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axis.legend(frameon=False, loc="upper left")
    generated["quality_runtime"] = _save(figure, output, "quality_runtime")
    plt.close(figure)

    partitions = sorted({record["partition"] for record in records})
    width = 0.8 / len(method_ids)
    figure, axis = plt.subplots(figsize=(max(3.5, 0.7 * len(partitions)), 2.5))
    for method_index, method in enumerate(method_ids):
        values = []
        for partition in partitions:
            selected = [
                record
                for record in by_method[method]
                if record["partition"] == partition
            ]
            gaps = _available_values(selected, "gap_to_pool_best")
            values.append(float("nan") if not gaps else 100.0 * mean(gaps))
        positions = [
            index - 0.4 + width / 2 + method_index * width
            for index in range(len(partitions))
        ]
        axis.bar(positions, values, width=width, label=_display_name(method))
    axis.set_xticks(
        range(len(partitions)),
        [_display_name(partition) for partition in partitions],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Mean gap to pool best (%)")
    axis.grid(True, axis="y", linestyle=":", linewidth=0.5)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        ncol=min(3, len(method_ids)),
    )
    figure.subplots_adjust(top=0.82, bottom=0.28)
    generated["gap_by_partition"] = _save(figure, output, "gap_by_partition")
    plt.close(figure)

    figure, axis = plt.subplots()
    bottoms = [0.0] * len(method_ids)
    for metric, label in TIMING_STACK:
        values = [
            mean(float(record["metrics"].get(metric, 0.0)) for record in by_method[method])
            for method in method_ids
        ]
        axis.bar(method_ids, values, bottom=bottoms, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_xticks(
        range(len(method_ids)),
        [_display_name(method) for method in method_ids],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Mean online solving time (s)")
    axis.grid(True, axis="y", linestyle=":", linewidth=0.5)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        ncol=3,
    )
    figure.subplots_adjust(top=0.76, bottom=0.30)
    generated["timing_decomposition"] = _save(
        figure, output, "timing_decomposition"
    )
    plt.close(figure)

    payload = {
        "schema_version": "1.0",
        "run_directory": str(run.resolve()),
        "figures": generated,
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
