"""Export the three-neural-generator forward-budget comparison figure.

The figure combines the Direct/Masked budget calibration evidence with the
Sequential-GNN calibration evidence and writes both a plotting-data JSON file
and publication-ready PDF/PNG figures for Section V.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT.parent
ARTIFACT_ROOT = ROOT / "artifacts"
OUT_ROOT = ARTIFACT_ROOT / "phase6f-budget-comparison"
FIGURE_DIR = PAPER_ROOT / "latex file" / "figures"

DIRECT_MASKED_EVIDENCE = (
    ARTIFACT_ROOT
    / "phase6f-forward-budget-calibration"
    / "forward_budget_calibration_evidence.json"
)
SEQUENTIAL_EVIDENCE = (
    ARTIFACT_ROOT
    / "phase6f-sequential-forward-budget-calibration"
    / "sequential_forward_budget_evidence.json"
)

METHODS = (
    ("Direct", "direct", "#6B7280", "o"),
    ("Sequential GNN", "sequential", "#2563EB", "s"),
    ("Masked Diffusion", "masked", "#D97706", "D"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(
    *,
    budget: int,
    label: str,
    family: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "budget": int(budget),
        "method": label,
        "family": family,
        "verified_proposal_percent": 100.0 * float(payload["verified_proposal_rate"]),
        "fallback_percent": 100.0 * float(payload["fallback_invocation_rate"]),
        "reference_gap_percent": 100.0 * float(payload["mean_gap_to_pool_best"]),
        "time_seconds": float(payload["mean_total_seconds"]),
        "realized_budget": float(payload["mean_realized_budget"]),
    }


def collect() -> list[dict[str, Any]]:
    direct_masked = _read_json(DIRECT_MASKED_EVIDENCE)
    sequential = _read_json(SEQUENTIAL_EVIDENCE)
    budgets = [int(value) for value in direct_masked["budgets"]]
    if budgets != [int(value) for value in sequential["budgets"]]:
        raise ValueError("Direct/Masked and Sequential budget grids do not match.")

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        rows.append(
            _row(
                budget=budget,
                label="Direct",
                family="direct",
                payload=direct_masked["overall"][f"direct_b{budget}"],
            )
        )
        rows.append(
            _row(
                budget=budget,
                label="Sequential GNN",
                family="sequential",
                payload=sequential["overall"][f"sequential_b{budget}"],
            )
        )
        rows.append(
            _row(
                budget=budget,
                label="Masked Diffusion",
                family="masked",
                payload=direct_masked["overall"][f"masked_b{budget}"],
            )
        )
    return rows


def _series(rows: list[dict[str, Any]], method: str, key: str) -> list[float]:
    return [float(row[key]) for row in rows if row["method"] == method]


def plot(rows: list[dict[str, Any]]) -> None:
    budgets = sorted({int(row["budget"]) for row in rows})
    x = np.arange(len(budgets))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.2))
    panels = (
        ("verified_proposal_percent", "Verified proposals (%)", "Proposal coverage"),
        ("reference_gap_percent", "Gap (%)", "Latency quality"),
        ("fallback_percent", "Fallback usage (%)", "Fallback burden"),
    )
    for axis, (metric, ylabel, subtitle) in zip(axes, panels):
        for label, _family, color, marker in METHODS:
            axis.plot(
                x,
                _series(rows, label, metric),
                label=label,
                color=color,
                marker=marker,
                markersize=4.2,
                linewidth=1.75,
                markeredgecolor="white",
                markeredgewidth=0.55,
            )
        axis.set_xticks(x, [str(value) for value in budgets])
        axis.set_xlabel(r"$B_{\mathrm{NN}}$")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="major", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_title(subtitle, pad=4)
        for spine in axis.spines.values():
            spine.set_color("#4B5563")
            spine.set_linewidth(0.8)

    axes[0].set_ylim(35, 101)
    axes[1].set_ylim(1.8, 5.7)
    axes[2].set_ylim(0, 61)
    fig.legend(
        handles=axes[0].lines,
        labels=[line.get_label() for line in axes[0].lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.21, top=0.72, wspace=0.46)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUT_ROOT / "stage43_forward_budget_three_algorithms_data.json"
    data_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"stage43_forward_budget_three_algorithms.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def main() -> None:
    rows = collect()
    plot(rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "data": str(OUT_ROOT / "stage43_forward_budget_three_algorithms_data.json"),
                "pdf": str(FIGURE_DIR / "stage43_forward_budget_three_algorithms.pdf"),
                "png": str(FIGURE_DIR / "stage43_forward_budget_three_algorithms.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
