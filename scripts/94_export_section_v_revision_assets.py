"""Export Section V evidence summaries and revision figures.

This script is intentionally read-only with respect to experiment artifacts.
It aggregates the frozen evaluation records used by the manuscript and writes
compact figure/data products for the LaTeX project.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LATEX_FIG_DIR = ROOT / "latex file" / "figures"
EVIDENCE_DIR = ROOT / "implementation" / "artifacts" / "paper-evidence-section-v-20260702"

NO_REPAIR_RECORDS = ROOT / "implementation" / "artifacts" / "phase6f-no-repair-neural-proposals" / "records"
SEQ_RECORDS = ROOT / "implementation" / "artifacts" / "phase6f-sequential-multiseed-evaluation" / "records"
LATEST_TABLES = (
    ROOT
    / "implementation"
    / "artifacts"
    / "phase6f-sequential-multiseed-evaluation"
    / "section_v_latest_tables.json"
)
BUDGET_DATA = (
    ROOT
    / "implementation"
    / "artifacts"
    / "phase6f-budget-comparison"
    / "stage43_forward_budget_three_algorithms_data.json"
)
CROSS_SCALE_DATA = (
    ROOT
    / "implementation"
    / "artifacts"
    / "phase6f-decoding-policy-full"
    / "cross_scale"
    / "decoding_enhancement_probe_evidence.json"
)


METHOD_LABELS = {
    "fallback_only": "Fallback construction",
    "direct_k64": "Direct GNN",
    "sequential_kseq": "Sequential GNN",
    "masked_deterministic_k1": "Masked deterministic",
    "masked_diffusion_k8": "Masked Diffusion",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return 100.0 * value


def aggregate_verify_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    return {
        "records": len(rows),
        "success_percent": pct(sum(bool(r.get("success")) for r in rows) / len(rows)),
        "raw_success_percent": pct(
            sum(bool(r.get("raw_any_feasible")) for r in rows) / len(rows)
        ),
        "proposal_feasible_percent": (
            pct(
                sum((r.get("raw_feasible_rate") or 0.0) for r in rows)
                / len(rows)
            )
            if any(r.get("raw_feasible_rate") is not None for r in rows)
            else None
        ),
        "fallback_percent": pct(sum(bool(r.get("fallback_invoked")) for r in rows) / len(rows)),
        "gap_percent": pct(sum(float(r.get("gap_to_pool_best", 0.0)) for r in rows) / len(rows)),
        "time_seconds": sum(float(r.get("total_seconds", 0.0)) for r in rows) / len(rows),
    }


def aggregate_main_evaluation() -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in (NO_REPAIR_RECORDS / "sealed_id").rglob("*.json"):
        record = load_json(path)
        for method, payload in record["methods"].items():
            if method == "random_k64":
                continue
            grouped.setdefault(method, []).append(payload["verify_fallback"])

    seq_rows = []
    for path in (SEQ_RECORDS / "sealed_id").rglob("*.json"):
        record = load_json(path)
        seq_rows.append(record["methods"]["sequential_kseq"])
    grouped["sequential_kseq"] = seq_rows

    ordered = [
        "fallback_only",
        "direct_k64",
        "sequential_kseq",
        "masked_deterministic_k1",
        "masked_diffusion_k8",
    ]
    return {method: aggregate_verify_rows(grouped[method]) for method in ordered}


def collect_generalization() -> dict[str, dict[str, dict[str, float | int]]]:
    latest = load_json(LATEST_TABLES)["by_setting"]
    cross = load_json(CROSS_SCALE_DATA)["overall"]
    groups: dict[str, dict[str, dict[str, float | int]]] = {
        "Main evaluation": {
            "Direct GNN": latest["sealed_id"]["direct_k64"],
            "Sequential GNN": latest["sealed_id"]["sequential_kseq"],
            "Masked Diffusion": latest["sealed_id"]["masked_diffusion_k8"],
        },
        "Controlled shift": {
            "Direct GNN": latest["controlled_shift"]["direct_k64"],
            "Sequential GNN": latest["controlled_shift"]["sequential_kseq"],
            "Masked Diffusion": latest["controlled_shift"]["masked_diffusion_k8"],
        },
        "Cross-scale": {
            "Direct GNN": {
                "verified_proposal_percent": pct(cross["direct_b64_t1"]["raw_any_feasible_rate"]),
                "final_gap_percent": pct(cross["direct_b64_t1"]["mean_gap"]),
                "time_seconds": cross["direct_b64_t1"]["mean_total_seconds"],
                "records": cross["direct_b64_t1"]["records"],
            },
            "Sequential GNN": {
                "verified_proposal_percent": pct(cross["sequential_b64_t1"]["raw_any_feasible_rate"]),
                "final_gap_percent": pct(cross["sequential_b64_t1"]["mean_gap"]),
                "time_seconds": cross["sequential_b64_t1"]["mean_total_seconds"],
                "records": cross["sequential_b64_t1"]["records"],
            },
            "Masked Diffusion": {
                "verified_proposal_percent": pct(cross["masked_k8_t1"]["raw_any_feasible_rate"]),
                "final_gap_percent": pct(cross["masked_k8_t1"]["mean_gap"]),
                "time_seconds": cross["masked_k8_t1"]["mean_total_seconds"],
                "records": cross["masked_k8_t1"]["records"],
            },
        },
        "Realistic simulation": {
            "Direct GNN": latest["realistic_profile"]["direct_k64"],
            "Sequential GNN": latest["realistic_profile"]["sequential_kseq"],
            "Masked Diffusion": latest["realistic_profile"]["masked_diffusion_k8"],
        },
    }
    return groups


def plot_budget() -> None:
    rows = load_json(BUDGET_DATA)["rows"]
    methods = ["Direct", "Sequential GNN", "Masked Diffusion"]
    colors = {
        "Direct": "#4C78A8",
        "Sequential GNN": "#F58518",
        "Masked Diffusion": "#54A24B",
    }
    markers = {"Direct": "o", "Sequential GNN": "s", "Masked Diffusion": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    for method in methods:
        method_rows = sorted([r for r in rows if r["method"] == method], key=lambda r: r["budget"])
        label = "Direct GNN" if method == "Direct" else method
        budgets = [r["budget"] for r in method_rows]
        gaps = [r["reference_gap_percent"] for r in method_rows]
        times = [r["time_seconds"] for r in method_rows]
        axes[0].plot(
            budgets,
            gaps,
            label=label,
            color=colors[method],
            marker=markers[method],
            linewidth=1.9,
            markersize=5.2,
        )
        axes[1].plot(
            budgets,
            times,
            label=label,
            color=colors[method],
            marker=markers[method],
            linewidth=1.9,
            markersize=5.2,
        )

    axes[0].set_xscale("log", base=2)
    axes[1].set_xscale("log", base=2)
    for ax in axes:
        ax.set_xticks([8, 16, 32, 64, 128])
        ax.set_xticklabels(["8", "16", "32", "64", "128"])
        ax.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(r"Forward-equivalent budget $B_{\mathrm{NN}}$")

    axes[0].set_ylabel("Gap (%)")
    axes[1].set_ylabel("Online time (s)")
    axes[0].legend(loc="upper right", frameon=True, fontsize=8)

    for ext in ("pdf", "png"):
        fig.savefig(LATEX_FIG_DIR / f"section_v_budget_quality_time.{ext}", dpi=300)
    plt.close(fig)


def plot_generalization(groups: dict[str, dict[str, dict[str, float | int]]]) -> None:
    method_names = ["Direct GNN", "Sequential GNN", "Masked Diffusion"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    labels = [
        group
        for group in ("Main evaluation", "Controlled shift", "Cross-scale")
        if group in groups
    ]
    x = np.arange(len(labels))
    width = 0.23

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.8), constrained_layout=False)
    for i, method in enumerate(method_names):
        raw_values = [
            groups[group][method]["verified_proposal_percent"] for group in labels
        ]
        gap_values = [groups[group][method]["final_gap_percent"] for group in labels]
        axes[0].bar(
            x + (i - 1) * width,
            raw_values,
            width,
            label=method,
            color=colors[i],
            edgecolor="#333333",
            linewidth=0.45,
        )
        axes[1].bar(
            x + (i - 1) * width,
            gap_values,
            width,
            label=method,
            color=colors[i],
            edgecolor="#333333",
            linewidth=0.45,
        )

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Main\nevaluation", "Controlled\nshift", "Cross-scale\nstress"],
            fontsize=8,
        )
        ax.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Raw-success (%)")
    axes[1].set_ylabel("Gap (%)")
    axes[0].set_ylim(0, 105)
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

    for ext in ("pdf", "png"):
        fig.savefig(LATEX_FIG_DIR / f"section_v_generalization_cross_scale.{ext}", dpi=300)
    plt.close(fig)


def write_evidence(main_eval: dict[str, Any], generalization: dict[str, Any]) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "section_v_revision_20260702",
        "decision": {
            "main_configuration": "Masked Diffusion K=8 / B_NN=64",
            "budget_128_role": "quality-time sensitivity point rather than the default comparison setting",
        },
        "main_evaluation": main_eval,
        "generalization_groups": generalization,
        "sources": {
            "main_direct_masked_fallback": str(NO_REPAIR_RECORDS.relative_to(ROOT)),
            "main_sequential": str(SEQ_RECORDS.relative_to(ROOT)),
            "latest_tables": str(LATEST_TABLES.relative_to(ROOT)),
            "budget_data": str(BUDGET_DATA.relative_to(ROOT)),
            "cross_scale_probe": str(CROSS_SCALE_DATA.relative_to(ROOT)),
        },
        "figures": [
            "latex file/figures/section_v_budget_quality_time.pdf",
            "latex file/figures/section_v_generalization_cross_scale.pdf",
        ],
    }
    (EVIDENCE_DIR / "section_v_revision_evidence.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Section V Revision Evidence Package",
        "",
        "This folder collects the compact evidence used by the July 2 Section V revision.",
        "",
        "## Default Configuration Decision",
        "",
        "- The manuscript keeps `Masked Diffusion K=8` with `B_NN=64` as the main setting.",
        "- `B_NN=128` is reported as a budget-sensitivity point rather than the default comparison setting.",
        "",
        "## Source Artifacts",
    ]
    for name, path in payload["sources"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Exported Figures",
            "",
            "- `latex file/figures/section_v_budget_quality_time.pdf`",
            "- `latex file/figures/section_v_generalization_cross_scale.pdf`",
        ]
    )
    (EVIDENCE_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    main_eval = aggregate_main_evaluation()
    generalization = collect_generalization()
    plot_budget()
    plot_generalization(generalization)
    write_evidence(main_eval, generalization)
    print(json.dumps({"main_evaluation": main_eval, "evidence_dir": str(EVIDENCE_DIR)}, indent=2))


if __name__ == "__main__":
    main()
