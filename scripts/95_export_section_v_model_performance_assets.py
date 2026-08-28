"""Export Section V-B model-performance assets.

This script aggregates frozen training logs and main-evaluation records for the
current absorbing-MASK manuscript branch. It writes a training-curve figure and
paired comparison statistics used by Section V-B.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LATEX_FIG_DIR = ROOT / "latex file" / "figures"
EVIDENCE_DIR = ROOT / "implementation" / "artifacts" / "paper-evidence-section-v-20260702"

DIRECT_MASKED_TRAIN = (
    ROOT / "implementation" / "artifacts" / "phase6e-e-stage39-10seed-training"
)
DIRECT_MASKED_EARLY_ROOTS = {
    2026070111: ROOT
    / "implementation"
    / "artifacts"
    / "phase6e-e-stage3-training",
    2026070112: ROOT
    / "implementation"
    / "artifacts"
    / "phase6e-e-stage38-training",
    2026070113: ROOT
    / "implementation"
    / "artifacts"
    / "phase6e-e-stage38-training",
}
SEQUENTIAL_TRAIN = (
    ROOT / "implementation" / "artifacts" / "phase6f-sequential-conditional-training"
)
NO_REPAIR_RECORDS = (
    ROOT
    / "implementation"
    / "artifacts"
    / "phase6f-no-repair-neural-proposals"
    / "records"
    / "sealed_id"
)
SEQ_RECORDS = (
    ROOT
    / "implementation"
    / "artifacts"
    / "phase6f-sequential-multiseed-evaluation"
    / "records"
    / "sealed_id"
)

COMMON_SEEDS = tuple(range(2026070111, 2026070121))

MODEL_SPECS = {
    "Direct GNN": {
        "root": DIRECT_MASKED_TRAIN,
        "dirname": "direct-seed{seed}",
        "loss_key": "loss_clean_state",
        "acc_key": "clean_accuracy",
        "color": "#4C78A8",
        "marker": "o",
    },
    "Sequential GNN": {
        "root": SEQUENTIAL_TRAIN,
        "dirname": "sequential_conditional-seed{seed}",
        "loss_key": "loss_next_reconstruction",
        "acc_key": "next_accuracy",
        "color": "#F58518",
        "marker": "s",
    },
    "Masked Diffusion": {
        "root": DIRECT_MASKED_TRAIN,
        "dirname": "masked_conditional-seed{seed}",
        "loss_key": "loss_masked_reconstruction",
        "acc_key": "hidden_accuracy",
        "color": "#54A24B",
        "marker": "^",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def collect_training_curves() -> dict[str, dict[str, list[float]]]:
    curves: dict[str, dict[str, list[float]]] = {}
    for method, spec in MODEL_SPECS.items():
        by_step: dict[int, list[tuple[float, float]]] = defaultdict(list)
        used_seeds: list[int] = []
        for seed in COMMON_SEEDS:
            root = spec["root"]
            if method != "Sequential GNN":
                root = DIRECT_MASKED_EARLY_ROOTS.get(seed, root)
            path = root / spec["dirname"].format(seed=seed) / "metrics.jsonl"
            if not path.exists():
                continue
            used_seeds.append(seed)
            for row in load_jsonl(path):
                if row.get("type") != "train":
                    continue
                step = int(float(row["step"]))
                loss = float(row[spec["loss_key"]])
                acc = float(row[spec["acc_key"]])
                by_step[step].append((loss, acc))

        steps = sorted(step for step, vals in by_step.items() if len(vals) == len(used_seeds))
        loss_mean = np.array([np.mean([v[0] for v in by_step[step]]) for step in steps])
        acc_mean = np.array([np.mean([v[1] for v in by_step[step]]) for step in steps])
        curves[method] = {
            "seeds": used_seeds,
            "steps": [float(step) for step in steps],
            "loss": [float(v) for v in smooth(loss_mean)],
            "accuracy_percent": [float(v * 100.0) for v in smooth(acc_mean)],
        }
    return curves


def plot_training_curves(curves: dict[str, dict[str, list[float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    for method, spec in MODEL_SPECS.items():
        curve = curves[method]
        steps = np.array(curve["steps"], dtype=np.float64) / 1000.0
        axes[0].plot(
            steps,
            curve["loss"],
            label=method,
            color=spec["color"],
            linewidth=1.85,
        )
        axes[1].plot(
            steps,
            curve["accuracy_percent"],
            label=method,
            color=spec["color"],
            linewidth=1.85,
        )

    axes[0].set_ylabel("Training CE loss")
    axes[1].set_ylabel("Target accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Training steps (k)")
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper right", frameon=True, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(LATEX_FIG_DIR / f"section_v_training_curves.{ext}", dpi=300)
    plt.close(fig)


def collect_paired_records() -> dict[tuple[int, str], dict[str, float]]:
    records: dict[tuple[int, str], dict[str, float]] = {}
    for path in NO_REPAIR_RECORDS.rglob("*.json"):
        record = load_json(path)
        seed = int(record["training_seed"])
        instance = str(record["instance_id"])
        methods = record["methods"]
        records[(seed, instance)] = {
            "Direct GNN": 100.0
            * float(methods["direct_k64"]["verify_fallback"]["gap_to_pool_best"]),
            "Masked Diffusion": 100.0
            * float(methods["masked_diffusion_k8"]["verify_fallback"]["gap_to_pool_best"]),
        }

    for path in SEQ_RECORDS.rglob("*.json"):
        record = load_json(path)
        seed = int(record["seed"])
        instance = str(record["instance_id"])
        key = (seed, instance)
        if key not in records:
            continue
        records[key]["Sequential GNN"] = 100.0 * float(
            record["methods"]["sequential_kseq"]["gap_to_pool_best"]
        )
    return {key: value for key, value in records.items() if len(value) == 3}


def binomial_greater_pvalue(successes: int, failures: int) -> float:
    n = successes + failures
    if n == 0:
        return 1.0
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        - n * math.log(2.0)
        for i in range(successes, n + 1)
    ]
    max_log = max(logs)
    return float(math.exp(max_log) * sum(math.exp(v - max_log) for v in logs))


def paired_tests(records: dict[tuple[int, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline in ("Direct GNN", "Sequential GNN"):
        lower = higher = ties = 0
        deltas: list[float] = []
        for payload in records.values():
            masked = payload["Masked Diffusion"]
            base = payload[baseline]
            delta = base - masked
            deltas.append(delta)
            if masked < base - 1e-10:
                lower += 1
            elif masked > base + 1e-10:
                higher += 1
            else:
                ties += 1
        rows.append(
            {
                "comparison": f"Masked Diffusion vs. {baseline}",
                "records": len(records),
                "masked_lower": lower,
                "masked_higher": higher,
                "ties": ties,
                "mean_gap_reduction_points": float(np.mean(deltas)),
                "one_sided_sign_p": binomial_greater_pvalue(lower, higher),
            }
        )
    return rows


def main() -> None:
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    curves = collect_training_curves()
    plot_training_curves(curves)
    records = collect_paired_records()
    paired = paired_tests(records)

    payload = {
        "scope": "section_v_b_model_performance_revision",
        "training_curve": {
            "common_seeds": list(COMMON_SEEDS),
            "source_roots": {
                "direct_masked": [
                    str(path.relative_to(ROOT))
                    for path in (
                        *dict.fromkeys(DIRECT_MASKED_EARLY_ROOTS.values()),
                        DIRECT_MASKED_TRAIN,
                    )
                ],
                "sequential": str(SEQUENTIAL_TRAIN.relative_to(ROOT)),
            },
            "curves": curves,
            "figure": "latex file/figures/section_v_training_curves.pdf",
        },
        "paired_tests": {
            "record_count": len(records),
            "sources": {
                "direct_masked": str(NO_REPAIR_RECORDS.relative_to(ROOT)),
                "sequential": str(SEQ_RECORDS.relative_to(ROOT)),
            },
            "rows": paired,
        },
    }
    output_path = EVIDENCE_DIR / "section_v_b_model_performance_evidence.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "paired_tests": paired}, indent=2))


if __name__ == "__main__":
    main()
