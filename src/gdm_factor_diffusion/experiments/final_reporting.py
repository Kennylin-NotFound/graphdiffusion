"""Generate Phase 6D-C paper evidence only from the frozen final campaign."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from gdm_factor_diffusion.common.logging import write_json

from .schema import file_sha256

STOCHASTIC_METHODS = (
    "diffusion_hybrid",
    "direct_hybrid",
    "random_hybrid",
    "fallback_only",
)
OPTIMIZATION_METHODS = ("milp_2s", "greedy_local", "fallback_only")
METHOD_NAMES = {
    "diffusion_hybrid": "Diffusion hybrid",
    "direct_hybrid": "Direct predictor hybrid",
    "random_hybrid": "Random hybrid",
    "fallback_only": "Fallback",
    "milp_2s": "MILP (2 s)",
    "greedy_local": "Greedy local",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_hash(path: str | Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError(f"Frozen evidence hash mismatch: {path}")


def verify_phase6d_c_evidence(path: str | Path) -> dict[str, Any]:
    """Verify the final campaign freeze and every artifact it references."""

    evidence = _read_json(path)
    if evidence.get("scope") != "phase6d_c_final_evidence":
        raise ValueError("Unsupported final evidence scope.")
    _verify_hash(evidence["campaign_lock"], evidence["campaign_lock_sha256"])
    _verify_hash(evidence["run_index"], evidence["run_index_sha256"])
    for aggregate in evidence["aggregates"].values():
        _verify_hash(aggregate["path"], aggregate["sha256"])
    for run in evidence["runs"]:
        root = Path(run["run_directory"])
        for filename, hash_key in (
            ("summary.json", "summary_sha256"),
            ("resolved_manifest.json", "resolved_manifest_sha256"),
            ("records.jsonl", "records_sha256"),
        ):
            _verify_hash(root / filename, run[hash_key])
    return evidence


def _stat(block: dict[str, Any], metric: str, statistic: str = "mean") -> float | None:
    values = block.get(metric)
    return None if values is None else float(values[statistic])


def _scaled_stat(
    block: dict[str, Any],
    metric: str,
    statistic: str = "mean",
    *,
    scale: float = 1.0,
) -> float | None:
    value = _stat(block, metric, statistic)
    return None if value is None else scale * value


def _stochastic_row(
    dataset: str,
    method: str,
    metrics: dict[str, Any],
    *,
    partition: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "partition": partition,
        "method_id": method,
        "method": METHOD_NAMES[method],
        "gap_percent_mean": _scaled_stat(metrics, "gap_to_pool_best", scale=100.0),
        "gap_percent_std": _scaled_stat(
            metrics, "gap_to_pool_best", "std", scale=100.0
        ),
        "success_percent_mean": _scaled_stat(metrics, "success", scale=100.0),
        "success_percent_std": _scaled_stat(metrics, "success", "std", scale=100.0),
        "raw_feasible_percent_mean": _scaled_stat(
            metrics, "raw_feasible_rate", scale=100.0
        ),
        "raw_feasible_percent_std": _scaled_stat(
            metrics, "raw_feasible_rate", "std", scale=100.0
        ),
        "total_seconds_mean": _stat(metrics, "total_seconds"),
        "total_seconds_std": _stat(metrics, "total_seconds", "std"),
        "repair_success_rate_mean": _stat(metrics, "repair_success_rate"),
        "fallback_invoked_rate_mean": _stat(metrics, "fallback_invoked"),
    }


def _stochastic_rows(dataset: str, aggregate: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall = [
        _stochastic_row(dataset, method, aggregate["methods"][method])
        for method in STOCHASTIC_METHODS
    ]
    partitions = [
        _stochastic_row(dataset, method, metrics, partition=partition)
        for partition, methods in aggregate["partitions"].items()
        for method, metrics in methods.items()
    ]
    return overall, partitions


def _optimization_rows(dataset: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in OPTIMIZATION_METHODS:
        block = summary["aggregate"]["methods"][method]
        metrics = block["metrics"]
        rows.append(
            {
                "dataset": dataset,
                "method_id": method,
                "method": METHOD_NAMES[method],
                "instances": int(block["instances"]),
                "successes": int(block["successes"]),
                "failures": int(block["failures"]),
                "success_percent": 100.0 * float(_stat(metrics, "success") or 0.0),
                "gap_percent_mean_conditional": 100.0
                * float(_stat(metrics, "gap_to_pool_best") or 0.0),
                "total_seconds_mean": float(_stat(metrics, "total_seconds") or 0.0),
                "milp_optimal_percent": 100.0
                * float(_stat(metrics, "milp_optimal") or 0.0),
                "milp_gap_mean": float(_stat(metrics, "milp_gap") or 0.0),
            }
        )
    return rows


def _pairwise_rows(dataset: str, aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair, metrics in aggregate["pairwise"].items():
        left, right = pair.split("__vs__")
        rows.append(
            {
                "dataset": dataset,
                "pair": pair,
                "left_method": METHOD_NAMES[left],
                "right_method": METHOD_NAMES[right],
                "both_success_mean_per_seed": _stat(metrics, "both_success"),
                "left_wins_mean_per_seed": _stat(metrics, "left_wins"),
                "ties_mean_per_seed": _stat(metrics, "ties"),
                "right_wins_mean_per_seed": _stat(metrics, "right_wins"),
                "left_only_success_mean_per_seed": _stat(metrics, "left_only_success"),
                "right_only_success_mean_per_seed": _stat(
                    metrics, "right_only_success"
                ),
                "both_failed_mean_per_seed": _stat(metrics, "both_failed"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
        }
    )
    return plt


def _save_figure(figure, output: Path, stem: str) -> list[str]:
    paths = []
    for suffix in (".png", ".pdf"):
        path = output / f"{stem}{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(str(path.resolve()))
    return paths


def _plot_quality_runtime(rows: list[dict[str, Any]], output: Path) -> list[str]:
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(3.5, 2.5))
    for row in rows:
        axis.errorbar(
            row["total_seconds_mean"],
            row["gap_percent_mean"],
            xerr=row["total_seconds_std"],
            yerr=row["gap_percent_std"],
            marker="o",
            linestyle="none",
            capsize=2,
            label=row["method"],
        )
    axis.set_xscale("log")
    axis.set_xlabel("Mean online solving time (s)")
    axis.set_ylabel("Mean gap to pool best (%)")
    axis.grid(True, which="both", linestyle=":", linewidth=0.5)
    axis.legend(frameon=False)
    paths = _save_figure(figure, output, "main_quality_runtime")
    plt.close(figure)
    return paths


def _plot_partition_metric(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    metric: str,
    ylabel: str,
    stem: str,
) -> list[str]:
    plt = _matplotlib()
    partitions = list(dict.fromkeys(str(row["partition"]) for row in rows))
    methods = [
        method
        for method in STOCHASTIC_METHODS
        if any(r["method_id"] == method and r[metric] is not None for r in rows)
    ]
    width = 0.8 / len(methods)
    figure, axis = plt.subplots(figsize=(max(3.5, 0.65 * len(partitions)), 2.5))
    for method_index, method in enumerate(methods):
        selected = {(str(row["partition"]), row["method_id"]): row for row in rows}
        values = [
            (
                float(selected[(partition, method)][metric])
                if selected[(partition, method)][metric] is not None
                else float("nan")
            )
            for partition in partitions
        ]
        positions = [
            index - 0.4 + width / 2 + method_index * width
            for index in range(len(partitions))
        ]
        axis.bar(positions, values, width=width, label=METHOD_NAMES[method])
    axis.set_xticks(
        range(len(partitions)),
        [partition.replace("_", " ") for partition in partitions],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", linestyle=":", linewidth=0.5)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
        ncol=2,
    )
    figure.subplots_adjust(top=0.80, bottom=0.31)
    paths = _save_figure(figure, output, stem)
    plt.close(figure)
    return paths


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def _format(value: float | None, digits: int) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _build_markdown(payload: dict[str, Any]) -> str:
    main = payload["stochastic"]["main"]["overall"]
    scale = payload["stochastic"]["scale"]["partitions"]
    main_opt = payload["optimization"]["main"]
    scale_opt = payload["optimization"]["scale"]
    main_by_method = {row["method_id"]: row for row in main}
    diffusion = main_by_method["diffusion_hybrid"]
    direct = main_by_method["direct_hybrid"]
    fallback = main_by_method["fallback_only"]
    quality_gain = 100.0 * (
        direct["gap_percent_mean"] - diffusion["gap_percent_mean"]
    ) / direct["gap_percent_mean"]
    speed_ratio = diffusion["total_seconds_mean"] / direct["total_seconds_mean"]

    lines = [
        "# Phase 6D-C Frozen Main-Comparison Results",
        "",
        "Generated only from the cryptographically frozen Phase 6D-C campaign.",
        "Exact objective gaps are conditional on successful outputs; success and",
        "failure rates are reported separately.",
        "",
        "## Main Five-Seed Stochastic Comparison",
        "",
        *_markdown_table(
            ["Method", "Gap (%)", "Raw feasible (%)", "Success (%)", "Online time (s)"],
            [
                [
                    row["method"],
                    f"{_format(row['gap_percent_mean'], 3)} +/- {_format(row['gap_percent_std'], 3)}",
                    (
                        "N/A"
                        if row["raw_feasible_percent_mean"] is None
                        else f"{_format(row['raw_feasible_percent_mean'], 2)} +/- "
                        f"{_format(row['raw_feasible_percent_std'], 2)}"
                    ),
                    _format(row["success_percent_mean"], 2),
                    f"{_format(row['total_seconds_mean'], 4)} +/- {_format(row['total_seconds_std'], 4)}",
                ]
                for row in main
            ],
        ),
        "",
        "Diffusion lowers the mean final gap by "
        f"{quality_gain:.1f}% relative to the matched direct predictor on the main",
        f"test/shift suite, while direct prediction is about {speed_ratio:.1f}x faster.",
        "All hybrid outputs pass final verification, but raw neural proposal",
        "feasibility remains far below 100%.",
        "",
        "## Controlled Shifts",
        "",
        *_markdown_table(
            ["Partition", "Diffusion gap (%)", "Direct gap (%)", "Fallback gap (%)"],
            [
                [
                    partition,
                    _format(methods["diffusion_hybrid"]["gap_percent_mean"], 3),
                    _format(methods["direct_hybrid"]["gap_percent_mean"], 3),
                    _format(methods["fallback_only"]["gap_percent_mean"], 3),
                ]
                for partition, methods in payload["stochastic"]["main"]["partition_map"].items()
            ],
        ),
        "",
        "## Scalability Boundary",
        "",
        *_markdown_table(
            ["Partition", "Diffusion gap (%)", "Direct gap (%)", "Raw feasible: diffusion/direct (%)"],
            [
                [
                    partition,
                    _format(methods["diffusion_hybrid"]["gap_percent_mean"], 3),
                    _format(methods["direct_hybrid"]["gap_percent_mean"], 3),
                    f"{_format(methods['diffusion_hybrid']['raw_feasible_percent_mean'], 2)} / "
                    f"{_format(methods['direct_hybrid']['raw_feasible_percent_mean'], 2)}",
                ]
                for partition, methods in payload["stochastic"]["scale"]["partition_map"].items()
            ],
        ),
        "",
        "Direct prediction is slightly better than diffusion on the large and",
        "extra-large scale partitions. Raw feasibility is near zero on large",
        "instances and zero for both learned methods on extra-large instances,",
        "so final feasibility comes from the declared repair/fallback/verifier",
        "pipeline.",
        "",
        "## Paired Main Outcomes",
        "",
        *_markdown_table(
            ["Pair", "Left wins", "Ties", "Right wins", "One-sided successes"],
            [
                [
                    f"{row['left_method']} vs {row['right_method']}",
                    _format(row["left_wins_mean_per_seed"], 1),
                    _format(row["ties_mean_per_seed"], 1),
                    _format(row["right_wins_mean_per_seed"], 1),
                    f"{_format(row['left_only_success_mean_per_seed'], 1)} / "
                    f"{_format(row['right_only_success_mean_per_seed'], 1)}",
                ]
                for row in payload["pairwise"]["main"]
            ],
        ),
        "",
        "## Optimization Baselines",
        "",
        *_markdown_table(
            ["Dataset", "Method", "Success", "Conditional gap (%)", "Online time (s)", "MILP optimal (%)"],
            [
                [
                    row["dataset"],
                    row["method"],
                    f"{row['successes']}/{row['instances']}",
                    f"{row['gap_percent_mean_conditional']:.3f}",
                    f"{row['total_seconds_mean']:.5f}",
                    f"{row['milp_optimal_percent']:.1f}",
                ]
                for row in main_opt + scale_opt
            ],
        ),
        "",
        "The two-second MILP proves optimality for every tested final instance and",
        "is faster than diffusion in the current synthetic regime. The present",
        "evidence therefore does not support a solver-superiority claim for the",
        "learned method. Greedy failures are retained rather than silently removed.",
        "",
        "## Evidence Boundary",
        "",
        "- Results apply to normalized synthetic instances under the direct-link,",
        "  no-runtime-contention formulation.",
        "- Fallback is deterministic; its repeated five-seed rows are paired anchors,",
        "  not evidence of stochastic variance.",
        "- Five-seed standard deviations summarize per-seed aggregate means, not",
        "  per-instance variability.",
        "- Final feasibility is guaranteed only for accepted outputs after hard",
        "  verification, not for raw neural proposals.",
        "- Phase 6E ablations and sensitivities remain required before Section V is",
        "  considered complete.",
        "",
    ]
    return "\n".join(lines)


def generate_phase6d_c_report(
    evidence_path: str | Path,
    *,
    output_directory: str | Path,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate auditable tables, figures, and interpretation from frozen evidence."""

    evidence = verify_phase6d_c_evidence(evidence_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    aggregates = {
        name: _read_json(record["path"])
        for name, record in evidence["aggregates"].items()
    }
    stochastic: dict[str, Any] = {}
    all_overall: list[dict[str, Any]] = []
    all_partitions: list[dict[str, Any]] = []
    pairwise = {}
    for dataset in ("main", "scale"):
        overall, partitions = _stochastic_rows(dataset, aggregates[dataset])
        stochastic[dataset] = {
            "overall": overall,
            "partitions": partitions,
            "partition_map": {
                partition: {
                    row["method_id"]: row
                    for row in partitions
                    if row["partition"] == partition
                }
                for partition in dict.fromkeys(str(row["partition"]) for row in partitions)
            },
        }
        all_overall.extend(overall)
        all_partitions.extend(partitions)
        pairwise[dataset] = _pairwise_rows(dataset, aggregates[dataset])

    optimization = {}
    for dataset in ("main", "scale"):
        run = next(
            record
            for record in evidence["runs"]
            if record["dataset"] == dataset and record["group"] == "optimization"
        )
        optimization[dataset] = _optimization_rows(
            dataset, _read_json(Path(run["run_directory"]) / "summary.json")
        )

    payload = {
        "schema_version": "1.0",
        "scope": "phase6d_c_frozen_result_report",
        "source_evidence": str(Path(evidence_path).resolve()),
        "source_evidence_sha256": file_sha256(evidence_path),
        "stochastic": stochastic,
        "pairwise": pairwise,
        "optimization": optimization,
    }
    report_json = write_json(output / "phase6d_c_results.json", payload)
    tables = {
        "stochastic_overall": str(
            _write_csv(output / "stochastic_overall.csv", all_overall).resolve()
        ),
        "stochastic_by_partition": str(
            _write_csv(output / "stochastic_by_partition.csv", all_partitions).resolve()
        ),
        "optimization": str(
            _write_csv(
                output / "optimization.csv",
                optimization["main"] + optimization["scale"],
            ).resolve()
        ),
        "stochastic_pairwise": str(
            _write_csv(
                output / "stochastic_pairwise.csv",
                pairwise["main"] + pairwise["scale"],
            ).resolve()
        ),
    }
    figures = {
        "main_quality_runtime": _plot_quality_runtime(
            stochastic["main"]["overall"], output
        ),
        "main_gap_by_partition": _plot_partition_metric(
            stochastic["main"]["partitions"],
            output,
            metric="gap_percent_mean",
            ylabel="Mean gap to pool best (%)",
            stem="main_gap_by_partition",
        ),
        "scale_gap_by_partition": _plot_partition_metric(
            stochastic["scale"]["partitions"],
            output,
            metric="gap_percent_mean",
            ylabel="Mean gap to pool best (%)",
            stem="scale_gap_by_partition",
        ),
        "scale_raw_feasibility": _plot_partition_metric(
            stochastic["scale"]["partitions"],
            output,
            metric="raw_feasible_percent_mean",
            ylabel="Mean raw feasible rate (%)",
            stem="scale_raw_feasibility",
        ),
    }
    markdown = Path(markdown_path) if markdown_path is not None else output / "PHASE6D_C_RESULTS.md"
    markdown.write_text(_build_markdown(payload), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "scope": "phase6d_c_result_report_manifest",
        "source_evidence": str(Path(evidence_path).resolve()),
        "source_evidence_sha256": file_sha256(evidence_path),
        "report_json": {"path": str(report_json.resolve()), "sha256": file_sha256(report_json)},
        "markdown": {"path": str(markdown.resolve()), "sha256": file_sha256(markdown)},
        "tables": {
            name: {"path": path, "sha256": file_sha256(path)}
            for name, path in tables.items()
        },
        "figures": {
            name: [
                {"path": path, "sha256": file_sha256(path)}
                for path in paths
            ]
            for name, paths in figures.items()
        },
    }
    write_json(output / "report_manifest.json", manifest)
    return manifest
