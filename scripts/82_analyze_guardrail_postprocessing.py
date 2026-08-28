"""Analyze failure-only fallback as a guardrail.

This script performs a post-hoc recomposition from existing per-instance
evaluation records. It does not retrain models or rerun neural inference.

Guardrail policy:
1. use the best raw/repaired verified candidate if repair-only succeeds;
2. invoke deterministic fallback only when repair-only has no feasible result.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts" / "phase6e-e-guardrail-postprocessing-analysis"

DATASETS = {
    "sealed_id": ROOT
    / "artifacts"
    / "phase6e-e-stage39-forward-budget-evaluation"
    / "records",
    "controlled_shift": ROOT
    / "artifacts"
    / "phase6e-e-controlled-shift-evaluation-10seed"
    / "records",
    "realistic_profile": ROOT
    / "artifacts"
    / "phase6e-e-realistic-profile-evaluation-10seed"
    / "records",
}

METHODS = (
    "direct_k64",
    "masked_deterministic_k1",
    "masked_diffusion_k8",
    "random_k64",
)
MAIN_DIRECT = "direct_k64"
MAIN_MASKED = "masked_diffusion_k8"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return mean(finite) if finite else None


def finite_std(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return pstdev(finite) if len(finite) > 1 else 0.0


def guardrail_payload(
    method_modes: Mapping[str, Any],
    fallback_payload: Mapping[str, Any],
) -> dict[str, Any]:
    repair = method_modes["repair_only"]
    if repair["success"]:
        payload = dict(repair)
        payload["guardrail_fallback_invoked"] = False
        payload["guardrail_rule"] = "repair_only_success"
        return payload

    payload = dict(fallback_payload)
    for key in (
        "raw_success",
        "raw_gap_to_pool_best",
        "pre_fallback_success",
        "pre_fallback_gap",
        "raw_any_feasible",
        "raw_feasible_count",
        "num_raw_proposals",
        "raw_feasible_rate",
        "raw_unique_rate",
        "raw_pairwise_hamming",
        "raw_capacity_violation_rate",
        "raw_link_violation_rate",
        "repair_attempts",
        "repair_successes",
        "repair_success_rate",
        "sampling_seconds",
        "repair_seconds",
    ):
        payload[key] = repair[key]
    payload["total_seconds"] = float(repair["total_seconds"]) + float(
        fallback_payload["total_seconds"]
    )
    payload["fallback_invoked"] = True
    payload["guardrail_fallback_invoked"] = True
    payload["guardrail_rule"] = "fallback_after_repair_failure"
    return payload


def aggregate(rows: list[Mapping[str, Any]], method_id: str, mode: str) -> dict[str, Any]:
    values = [row["methods"][method_id][mode] for row in rows]
    sources = ("raw", "repair", "fallback", "failure")
    return {
        "records": len(values),
        "success_rate": mean(float(row["success"]) for row in values),
        "mean_gap_to_pool_best": finite_mean([row["gap_to_pool_best"] for row in values]),
        "gap_std": finite_std([row["gap_to_pool_best"] for row in values]),
        "raw_success_rate": mean(float(row["raw_success"]) for row in values),
        "mean_raw_gap_to_pool_best": finite_mean(
            [row["raw_gap_to_pool_best"] for row in values]
        ),
        "pre_fallback_success_rate": mean(
            float(row["pre_fallback_success"]) for row in values
        ),
        "mean_pre_fallback_gap": finite_mean([row["pre_fallback_gap"] for row in values]),
        "fallback_invocation_rate": mean(
            float(row.get("guardrail_fallback_invoked", row["fallback_invoked"]))
            for row in values
        ),
        "repair_attempts_mean": mean(float(row["repair_attempts"]) for row in values),
        "repair_success_rate_mean": mean(
            float(row["repair_success_rate"]) for row in values
        ),
        "mean_total_seconds": mean(float(row["total_seconds"]) for row in values),
        "source_rates": {
            source: mean(float(row["source"] == source) for row in values)
            for source in sources
        },
    }


def score(rows: list[Mapping[str, Any]], method_id: str, mode: str) -> tuple[float, float]:
    values = [row["methods"][method_id][mode] for row in rows]
    success = mean(float(row["success"]) for row in values)
    gaps = [
        float(row["gap_to_pool_best"])
        for row in values
        if row["success"] and row["gap_to_pool_best"] is not None
    ]
    return (-success, mean(gaps) if gaps else float("inf"))


def paired(
    records: list[Mapping[str, Any]],
    *,
    left: tuple[str, str],
    right: tuple[str, str],
) -> dict[str, Any]:
    by_instance: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_instance[str(row["instance_id"])].append(row)
    left_wins = right_wins = ties = 0
    for rows in by_instance.values():
        left_score = score(rows, left[0], left[1])
        right_score = score(rows, right[0], right[1])
        if left_score < right_score:
            left_wins += 1
        elif right_score < left_score:
            right_wins += 1
        else:
            ties += 1
    n = left_wins + right_wins
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(left_wins, right_wins) + 1))
        p_value = min(1.0, 2.0 * tail / (2**n))
    return {
        "left": {"method": left[0], "mode": left[1]},
        "right": {"method": right[0], "mode": right[1]},
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
        "sign_test_pvalue": p_value,
    }


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.3f}%"


def analyze_dataset(dataset_id: str, records_root: Path) -> dict[str, Any]:
    records = [read_json(path) for path in sorted(records_root.rglob("*.json"))]
    if not records:
        raise FileNotFoundError(f"No records found under {records_root}")

    recomposed: list[dict[str, Any]] = []
    for record in records:
        methods = dict(record["methods"])
        for method_id in METHODS:
            if method_id not in methods:
                continue
            method_modes = dict(methods[method_id])
            if "repair_only" not in method_modes:
                continue
            fallback = methods.get("fallback_only", {}).get("full") or method_modes.get("full")
            if fallback is None:
                raise ValueError(
                    f"Missing fallback source for {dataset_id}/{method_id}/"
                    f"{record.get('instance_id')}"
                )
            method_modes["guardrail"] = guardrail_payload(method_modes, fallback)
            methods[method_id] = method_modes
        recomposed.append(
            {
                "training_seed": record["training_seed"],
                "instance_id": record["instance_id"],
                "pool_best": record["pool_best"],
                "methods": methods,
            }
        )

    modes = ("raw_only", "repair_only", "guardrail", "full")
    available_methods = tuple(method for method in METHODS if method in recomposed[0]["methods"])
    overall = {
        method_id: {mode: aggregate(recomposed, method_id, mode) for mode in modes}
        for method_id in available_methods
    }
    return {
        "records_root": str(records_root),
        "records": len(recomposed),
        "overall": overall,
        "paired": {
            "direct_vs_masked_guardrail": paired(
                recomposed,
                left=(MAIN_MASKED, "guardrail"),
                right=(MAIN_DIRECT, "guardrail"),
            ),
            "masked_guardrail_vs_full": paired(
                recomposed,
                left=(MAIN_MASKED, "guardrail"),
                right=(MAIN_MASKED, "full"),
            ),
            "masked_guardrail_vs_repair_only": paired(
                recomposed,
                left=(MAIN_MASKED, "guardrail"),
                right=(MAIN_MASKED, "repair_only"),
            ),
        },
    }


def write_report(path: Path, evidence: Mapping[str, Any]) -> None:
    sealed = evidence["datasets"]["sealed_id"]
    lines = [
        "# Guardrail Post-Processing Analysis",
        "",
        "Question: what happens if fallback is demoted from an always-included",
        "candidate to a failure-only guardrail?",
        "",
        "Guardrail policy: use the best raw/repaired verified candidate when",
        "available; invoke deterministic fallback only if repair-only has no",
        "feasible candidate.",
        "",
        "## Sealed ID Main Same-Budget Comparison",
        "",
        "| Method | Mode | Success | Ref. gap | Raw succ. | Pre-fallback succ. | Fallback invoked/selected | Source(raw/repair/fallback/failure) | Time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_id in (MAIN_DIRECT, MAIN_MASKED):
        for mode in ("raw_only", "repair_only", "guardrail", "full"):
            row = sealed["overall"][method_id][mode]
            source = row["source_rates"]
            lines.append(
                f"| {method_id} | {mode} | "
                f"{pct(row['success_rate'])} | "
                f"{pct(row['mean_gap_to_pool_best'])} | "
                f"{pct(row['raw_success_rate'])} | "
                f"{pct(row['pre_fallback_success_rate'])} | "
                f"{pct(row['fallback_invocation_rate'])} / {pct(source['fallback'])} | "
                f"{pct(source['raw'])}/{pct(source['repair'])}/{pct(source['fallback'])}/{pct(source['failure'])} | "
                f"{row['mean_total_seconds']:.3f}s |"
            )

    lines.extend(
        [
            "",
            "## Cross-Setting Summary",
            "",
            "| Setting | Method | Guardrail succ. | Guardrail ref. gap | Full ref. gap | Guardrail fallback selected | Full fallback selected |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_id, dataset in evidence["datasets"].items():
        for method_id in (MAIN_DIRECT, MAIN_MASKED):
            guardrail = dataset["overall"][method_id]["guardrail"]
            full = dataset["overall"][method_id]["full"]
            lines.append(
                f"| {dataset_id} | {method_id} | "
                f"{pct(guardrail['success_rate'])} | "
                f"{pct(guardrail['mean_gap_to_pool_best'])} | "
                f"{pct(full['mean_gap_to_pool_best'])} | "
                f"{pct(guardrail['source_rates']['fallback'])} | "
                f"{pct(full['source_rates']['fallback'])} |"
            )

    lines.extend(["", "## Paired Tests", ""])
    for dataset_id, dataset in evidence["datasets"].items():
        for name, result in dataset["paired"].items():
            lines.append(
                f"- {dataset_id}/{name}: left={result['left']['method']}:"
                f"{result['left']['mode']}, right={result['right']['method']}:"
                f"{result['right']['mode']}, wins/losses/ties="
                f"{result['left_wins']}/{result['right_wins']}/{result['ties']}, "
                f"p={result['sign_test_pvalue']:.6g}."
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- For Masked Diffusion, repair-only is already close to full feasibility.",
            "- Failure-only fallback therefore mainly handles a small residual failure set.",
            "- Always-included fallback strongly improves the final reference gap, but many selected outputs then come from deterministic fallback.",
            "- Guardrail mode better matches a manuscript story where feasibility is protected by the solver, while solution quality mostly reflects raw/repaired masked-diffusion candidates.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    datasets = {
        dataset_id: analyze_dataset(dataset_id, records_root)
        for dataset_id, records_root in DATASETS.items()
    }
    evidence = {
        "schema_version": "1.0",
        "scope": "phase6e_e_guardrail_postprocessing_analysis",
        "intervention": {
            "name": "guardrail",
            "definition": (
                "Use raw/repaired verified candidates when available; invoke "
                "fallback only if repair-only produces no feasible candidate."
            ),
        },
        "datasets": datasets,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "guardrail_postprocessing_evidence.json", evidence)
    write_report(OUTPUT_ROOT / "guardrail_postprocessing_report.md", evidence)
    print(
        json.dumps(
            {
                "output_root": str(OUTPUT_ROOT),
                "datasets": {
                    dataset_id: {
                        "records": dataset["records"],
                        "masked_guardrail_gap_percent": 100
                        * dataset["overall"][MAIN_MASKED]["guardrail"][
                            "mean_gap_to_pool_best"
                        ],
                        "masked_guardrail_success_percent": 100
                        * dataset["overall"][MAIN_MASKED]["guardrail"]["success_rate"],
                        "masked_guardrail_fallback_selected_percent": 100
                        * dataset["overall"][MAIN_MASKED]["guardrail"][
                            "source_rates"
                        ]["fallback"],
                    }
                    for dataset_id, dataset in datasets.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
