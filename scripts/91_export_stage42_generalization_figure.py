"""Export the Section V generalization figure with current terminology."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT.parent
SUMMARY = (
    ROOT
    / "artifacts"
    / "phase6f-sequential-multiseed-evaluation"
    / "section_v_stage42_summary.json"
)
FIGURE_DIR = PAPER_ROOT / "latex file" / "figures"

GROUPS = (
    ("ID holdout", "Evaluation\ndataset"),
    ("Controlled shift", "Controlled\nshift"),
    ("Profiled", "Profiled\nworkload"),
)
METHODS = (
    ("direct_k64", "Direct K=64", "#9EA7A6"),
    ("sequential_kseq", "Sequential GNN", "#F2B24A"),
    ("masked_diffusion_k8", "Masked Diffusion K=8", "#4A90E2"),
)


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    generalization = summary["generalization"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(GROUPS))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35))
    metrics = (
        ("verified_proposal_percent", "(a) Verified proposal coverage", "Verified proposal rate (%)", (0, 105)),
        ("reference_gap_percent", "(b) Gap", "Gap (%)", (0, 6.3)),
    )

    for axis, (metric, title, ylabel, ylim) in zip(axes, metrics):
        for offset, (method_key, label, color) in enumerate(METHODS):
            values = [
                float(generalization[group_key][method_key][metric])
                for group_key, _group_label in GROUPS
            ]
            axis.bar(
                x + (offset - 1) * width,
                values,
                width,
                label=label,
                color=color,
                edgecolor="#111827",
                linewidth=0.55,
                alpha=0.96,
            )
        axis.set_title(title, pad=4)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [label for _key, label in GROUPS])
        axis.set_ylim(*ylim)
        axis.grid(True, axis="y", linestyle="--", linewidth=0.55, color="#E5E7EB")
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("#111827")
            spine.set_linewidth(0.8)

    fig.legend(
        handles=[
            Patch(facecolor=color, edgecolor="#111827", label=label)
            for _key, label, color in METHODS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.4,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.73, wspace=0.45)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"stage42_neural_generalization.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)
    print(FIGURE_DIR / "stage42_neural_generalization.pdf")


if __name__ == "__main__":
    main()
