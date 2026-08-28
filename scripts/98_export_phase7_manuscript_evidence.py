"""Export manuscript tables and figures from the Phase 7 evidence freeze."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.experiments.schema import file_sha256


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = ROOT / "implementation"
ARTIFACTS = IMPLEMENTATION / "artifacts"
FIGURES = ROOT / "latex file" / "figures"
OUTPUT = ARTIFACTS / "paper-evidence-phase7-advisor-revision"

EVIDENCE_PATHS = {
    "main_cross": ARTIFACTS
    / "phase7-proposal-conditioned-recovery"
    / "phase7_proposal_recovery_evidence.json",
    "controlled": ARTIFACTS
    / "phase7-proposal-conditioned-recovery-controlled"
    / "phase7_proposal_recovery_evidence.json",
    "realistic": ARTIFACTS
    / "phase7-proposal-conditioned-recovery-realistic"
    / "phase7_proposal_recovery_evidence.json",
}

DIRECT_MASKED_BUDGET_RECORDS = (
    ARTIFACTS / "phase6f-forward-budget-calibration" / "records"
)
SEQUENTIAL_BUDGET_RECORDS = (
    ARTIFACTS / "phase6f-sequential-forward-budget-calibration" / "records"
)

BUDGETS = (8, 16, 32, 64, 128)
METHODS = ("Direct GNN", "Sequential GNN", "Masked Diffusion")
METHOD_IDS = {
    "Direct GNN": "direct_b64_t1",
    "Sequential GNN": "sequential_b64_t1",
    "Masked Diffusion": "masked_k8_t1",
}
COLORS = {
    "Direct GNN": "#4C78A8",
    "Sequential GNN": "#F28E2B",
    "Masked Diffusion": "#59A14F",
}
MARKERS = {"Direct GNN": "o", "Sequential GNN": "s", "Masked Diffusion": "^"}
DISPLAY_LABELS = {
    "random_k64": "Random + Recovery",
    "direct_b64_t1": "Direct GNN",
    "sequential_b64_t1": "Sequential GNN",
    "masked_k1_t1_mix": "Masked deterministic",
    "masked_k8_t1": "Masked Diffusion",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_budget_records() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for path in sorted(DIRECT_MASKED_BUDGET_RECORDS.rglob("*.json")):
        methods = load_json(path)["methods"]
        for budget in BUDGETS:
            grouped[("Direct GNN", budget)].append(methods[f"direct_b{budget}"])
            grouped[("Masked Diffusion", budget)].append(methods[f"masked_b{budget}"])
    for path in sorted(SEQUENTIAL_BUDGET_RECORDS.rglob("*.json")):
        methods = load_json(path)["methods"]
        for budget in BUDGETS:
            grouped[("Sequential GNN", budget)].append(
                methods[f"sequential_b{budget}"]
            )

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for budget in BUDGETS:
            records = grouped[(method, budget)]
            if not records:
                raise ValueError(f"Missing budget records for {method}, B_NN={budget}.")
            rows.append(
                {
                    "method": method,
                    "target_budget": budget,
                    "records": len(records),
                    "raw_success_rate": mean(
                        float(bool(record["raw_any_feasible"])) for record in records
                    ),
                    "mean_generation_seconds": mean(
                        float(record["sampling_seconds"]) for record in records
                    ),
                    "mean_realized_budget": mean(
                        float(record["realized_budget"]) for record in records
                    ),
                }
            )
    return rows


def _plot_budget(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    for method in METHODS:
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["target_budget"],
        )
        budgets = [row["target_budget"] for row in selected]
        raw_success = [100.0 * row["raw_success_rate"] for row in selected]
        generation_time = [row["mean_generation_seconds"] for row in selected]
        style = {
            "label": method,
            "color": COLORS[method],
            "marker": MARKERS[method],
            "linewidth": 1.9,
            "markersize": 5.2,
        }
        axes[0].plot(budgets, raw_success, **style)
        axes[1].plot(budgets, generation_time, **style)

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(BUDGETS)
        axis.set_xticklabels([str(value) for value in BUDGETS])
        axis.set_xlabel(r"Forward-equivalent budget $B_{\mathrm{NN}}$")
        axis.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Raw-success (%)")
    axes[0].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("Neural generation time (s)")
    axes[0].legend(loc="lower right", frameon=True, fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"section_v_budget_quality_time.{suffix}", dpi=300)
    plt.close(fig)


def _group(evidence: Mapping[str, Any], setting: str) -> dict[str, Any]:
    return {
        method: evidence["by_setting"][setting][method_id]
        for method, method_id in METHOD_IDS.items()
    }


def _plot_generalization(groups: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    labels = ("Main evaluation", "Controlled shift", "Cross-scale stress")
    x = np.arange(len(labels))
    width = 0.23
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.8), constrained_layout=False)
    for index, method in enumerate(METHODS):
        raw = [100.0 * groups[label][method]["raw_any_feasible_rate"] for label in labels]
        gap = [100.0 * groups[label][method]["mean_gap"] for label in labels]
        offset = x + (index - 1) * width
        axes[0].bar(
            offset,
            raw,
            width,
            label=method,
            color=COLORS[method],
            edgecolor="#333333",
            linewidth=0.45,
        )
        axes[1].bar(
            offset,
            gap,
            width,
            label=method,
            color=COLORS[method],
            edgecolor="#333333",
            linewidth=0.45,
        )

    tick_labels = ("Main\nevaluation", "Controlled\nshift", "Cross-scale\nstress")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(tick_labels, fontsize=8)
        axis.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Raw-success (%)")
    axes[0].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("Gap (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.82, wspace=0.25)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURES / f"section_v_generalization_cross_scale.{suffix}", dpi=300
        )
    plt.close(fig)


def _report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 7 Manuscript Evidence",
        "",
        "The main, controlled-shift, cross-scale, and realistic results use the",
        "verified-raw-then-proposal-recovery solver contract. Budget sensitivity",
        "uses raw proposal success and neural generation time, which are independent",
        "of post-generation recovery.",
        "",
        "## Evidence Sources",
        "",
    ]
    for name, path in payload["sources"].items():
        lines.append(f"- `{name}`: `{path}`")

    setting_names = (
        ("main", "Main Evaluation"),
        ("controlled_shift", "Controlled Shift"),
        ("cross_scale", "Cross-Scale Stress"),
        ("realistic_simulation", "Realistic Simulation"),
    )
    for key, title in setting_names:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Method | Records | Success | Raw success | Recovery source | Gap | Time (s) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method_id, row in payload[key].items():
            source_rates = row.get("source_rates", {})
            lines.append(
                "| {method} | {records} | {success:.3f}% | {raw:.3f}% | "
                "{recovery:.3f}% | {gap:.3f}% | {time:.3f} |".format(
                    method=DISPLAY_LABELS.get(method_id, method_id),
                    records=int(row["records"]),
                    success=100.0 * float(row["success_rate"]),
                    raw=100.0 * float(row["raw_any_feasible_rate"]),
                    recovery=100.0 * float(source_rates.get("recovery", 0.0)),
                    gap=100.0 * float(row["mean_gap"]),
                    time=float(row["mean_total_seconds"]),
                )
            )

    lines.extend(
        [
            "",
            "## Forward-Budget Raw Metrics",
            "",
            "| Method | Target B_NN | Records | Raw success | Neural generation time (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["budget_raw_metrics"]:
        lines.append(
            "| {method} | {budget} | {records} | {raw:.3f}% | {time:.3f} |".format(
                method=row["method"],
                budget=int(row["target_budget"]),
                records=int(row["records"]),
                raw=100.0 * float(row["raw_success_rate"]),
                time=float(row["mean_generation_seconds"]),
            )
        )
    lines.extend(["", "## Main Paired Tests", ""])
    for baseline, row in payload["main_paired_tests"].items():
        lines.append(
            f"- {baseline}: wins/losses/ties/skipped="
            f"{row['wins']}/{row['losses']}/{row['ties']}/{row['skipped']}, "
            f"mean Gap reduction={100.0 * row['mean_gap_reduction']:.3f} pp, "
            f"p={row['p_value_two_sided_sign_test']:.6g}."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    missing = [str(path) for path in EVIDENCE_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Phase 7 evidence is incomplete: " + ", ".join(missing))
    evidence = {name: load_json(path) for name, path in EVIDENCE_PATHS.items()}
    budget_rows = _collect_budget_records()
    groups = {
        "Main evaluation": _group(evidence["main_cross"], "main_evaluation"),
        "Controlled shift": _group(evidence["controlled"], "controlled_shift"),
        "Cross-scale stress": _group(evidence["main_cross"], "cross_scale"),
        "Realistic simulation": _group(
            evidence["realistic"], "realistic_simulation"
        ),
    }
    _plot_budget(budget_rows)
    _plot_generalization(groups)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "scope": "phase7_advisor_revision_manuscript_evidence",
        "main": evidence["main_cross"]["by_setting"]["main_evaluation"],
        "controlled_shift": evidence["controlled"]["by_setting"]["controlled_shift"],
        "cross_scale": evidence["main_cross"]["by_setting"]["cross_scale"],
        "realistic_simulation": evidence["realistic"]["by_setting"]["realistic_simulation"],
        "main_paired_tests": evidence["main_cross"]["paired_main_evaluation"],
        "budget_raw_metrics": budget_rows,
        "sources": {
            name: path.relative_to(ROOT).as_posix()
            for name, path in EVIDENCE_PATHS.items()
        },
    }
    summary_path = OUTPUT / "section_v_phase7_summary.json"
    report_path = OUTPUT / "section_v_phase7_report.md"
    write_json(summary_path, payload)
    report_path.write_text(_report(payload), encoding="utf-8")
    freeze = {
        "scope": "phase7_advisor_revision_manuscript_evidence_freeze",
        "summary": summary_path.relative_to(ROOT).as_posix(),
        "summary_sha256": file_sha256(summary_path),
        "report": report_path.relative_to(ROOT).as_posix(),
        "report_sha256": file_sha256(report_path),
        "source_sha256": {
            name: file_sha256(path) for name, path in EVIDENCE_PATHS.items()
        },
        "figures": {
            path.name: file_sha256(path)
            for path in (
                FIGURES / "section_v_budget_quality_time.pdf",
                FIGURES / "section_v_generalization_cross_scale.pdf",
            )
        },
    }
    write_json(OUTPUT / "evidence_freeze.json", freeze)
    print(json.dumps({"summary": str(summary_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
