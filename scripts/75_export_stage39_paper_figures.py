"""Export paper figures for the Stage 3.9 masked-diffusion evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "implementation"
    / "artifacts"
    / "phase6e-e-stage39-forward-budget-evaluation"
    / "forward_budget_evidence.json"
)
OUT_DIR = ROOT / "latex file" / "figures"


def _percent(value: float) -> float:
    return 100.0 * float(value)


def _load() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def export_budget_sensitivity(data: dict) -> None:
    budgets = [8, 16, 32, 64, 128]
    direct_gap = [
        _percent(data["overall"][f"direct_k{budget}"]["full"]["mean_gap_to_pool_best"])
        for budget in budgets
    ]
    masked_key = {
        8: "masked_deterministic_k1",
        16: "masked_diffusion_k2",
        32: "masked_diffusion_k4",
        64: "masked_diffusion_k8",
        128: "masked_diffusion_k16",
    }
    masked_gap = [
        _percent(data["overall"][masked_key[budget]]["full"]["mean_gap_to_pool_best"])
        for budget in budgets
    ]
    direct_raw = [
        _percent(data["overall"][f"direct_k{budget}"]["full"]["raw_success_rate"])
        for budget in budgets
    ]
    masked_raw = [
        _percent(data["overall"][masked_key[budget]]["full"]["raw_success_rate"])
        for budget in budgets
    ]

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35))

    axes[0].plot(budgets, direct_gap, marker="o", linewidth=1.5, label="Direct")
    axes[0].plot(
        budgets,
        masked_gap,
        marker="s",
        linewidth=1.5,
        label="Masked Diffusion",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(budgets, [str(b) for b in budgets])
    axes[0].set_xlabel(r"Neural forward-equivalent budget $B_{\mathrm{NN}}$")
    axes[0].set_ylabel("Final gap (%)")
    axes[0].grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(budgets, direct_raw, marker="o", linewidth=1.5, label="Direct")
    axes[1].plot(
        budgets,
        masked_raw,
        marker="s",
        linewidth=1.5,
        label="Masked Diffusion",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(budgets, [str(b) for b in budgets])
    axes[1].set_xlabel(r"Neural forward-equivalent budget $B_{\mathrm{NN}}$")
    axes[1].set_ylabel("Raw feasible rate (%)")
    axes[1].set_ylim(35, 100)
    axes[1].grid(True, linestyle="--", linewidth=0.4, alpha=0.55)
    axes[1].legend(frameon=False, loc="lower right")

    fig.tight_layout(w_pad=1.7)
    fig.savefig(OUT_DIR / "stage39_forward_budget_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def export_source_composition(data: dict) -> None:
    methods = [
        ("Fallback", "fallback_only"),
        ("Random K=64", "random_k64"),
        ("Direct K=64", "direct_k64"),
        ("Masked Det. K=1", "masked_deterministic_k1"),
        ("Masked Diff. K=8", "masked_diffusion_k8"),
    ]
    raw = [
        _percent(data["overall"][key]["full"]["source_rates"]["raw"])
        for _, key in methods
    ]
    repair = [
        _percent(data["overall"][key]["full"]["source_rates"]["repair"])
        for _, key in methods
    ]
    fallback = [
        _percent(data["overall"][key]["full"]["source_rates"]["fallback"])
        for _, key in methods
    ]
    labels = [name for name, _ in methods]

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    xs = range(len(methods))
    ax.bar(xs, raw, color="#2F6BBA", label="Raw")
    ax.bar(xs, repair, bottom=raw, color="#6ABF69", label="Repair")
    bottom = [r + p for r, p in zip(raw, repair)]
    ax.bar(xs, fallback, bottom=bottom, color="#F0A64A", label="Fallback")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Selected source (%)")
    ax.set_xticks(list(xs), labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.55)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "stage39_source_composition.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    export_budget_sensitivity(data)
    export_source_composition(data)
    print(f"Exported Stage 3.9 paper figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
