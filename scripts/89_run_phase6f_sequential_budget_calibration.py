"""Sequential-GNN forward-budget calibration for Phase 6F.

This script complements ``88_run_phase6f_forward_budget_calibration.py``.
The Direct and Masked-Diffusion budget sweep is already frozen there; here we
evaluate the trained Sequential Conditional GNN on the same ID holdout, seeds,
and target forward-equivalent budgets.  The policy is identical to the current
manuscript pipeline: verified generated proposals are selected by exact latency,
and the deterministic fallback is invoked only when no generated proposal is
verified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.experiments.schema import file_sha256
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    SequentialDecodeConfig,
    sample_sequential_proposals,
    solve_from_proposals,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze


SCOPE = "phase6f_sequential_forward_budget_calibration"
RECORD_SCOPE = f"{SCOPE}_record"
EVIDENCE_SCOPE = f"{SCOPE}_evidence"

DEFAULT_SEEDS = tuple(range(2026070111, 2026070121))
DEFAULT_BUDGETS = (8, 16, 32, 64, 128)
DATASET_ROOT = "artifacts/datasets/phase6e-e-stage38-sealed"
PARTITION = "sealed_test_id"
SEQUENTIAL = "sequential"


def _load_sequential_helpers() -> Any:
    path = Path(__file__).with_name("87_run_phase6f_sequential_multiseed_eval.py")
    spec = importlib.util.spec_from_file_location("phase6f_sequential_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Sequential helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SEQ = _load_sequential_helpers()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(root: Path, path: str | Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest().upper()


def _parse_int_csv(value: str | None, *, default: Sequence[int]) -> tuple[int, ...]:
    if value is None or not value.strip():
        return tuple(default)
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_seeds(value: str | None) -> tuple[int, ...]:
    seeds = _parse_int_csv(value, default=DEFAULT_SEEDS)
    unknown = sorted(set(seeds) - set(DEFAULT_SEEDS))
    if unknown:
        raise ValueError(f"Unsupported ten-seed freeze seeds: {unknown}")
    return tuple(sorted(dict.fromkeys(seeds)))


def _parse_budgets(value: str | None) -> tuple[int, ...]:
    budgets = _parse_int_csv(value, default=DEFAULT_BUDGETS)
    invalid = [budget for budget in budgets if budget < 1]
    if invalid:
        raise ValueError(f"Budgets must be positive: {invalid}")
    return tuple(sorted(dict.fromkeys(budgets)))


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _sequential_checkpoint(checkpoint_root: Path, seed: int) -> Path:
    path = checkpoint_root / f"sequential_conditional-seed{seed}" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Sequential checkpoint for seed {seed}: {path}")
    return path


def _protocol(
    *,
    root: Path,
    selected_seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    checkpoint_root: Path,
    output_root: Path,
    device: str,
    max_instances: int | None,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scope": SCOPE,
        "dataset_root": DATASET_ROOT,
        "partition": PARTITION,
        "selected_seeds": list(selected_seeds),
        "budgets": list(budgets),
        "sequential_checkpoint_root": _relative(root, checkpoint_root),
        "output_root": _relative(root, output_root),
        "device": device,
        "deterministic": True,
        "evaluation_seed": 2026071311,
        "sample_batch_size": 8,
        "temperature": 1.0,
        "max_instances": max_instances,
        "fallback_max_search_nodes": int(fallback_max_search_nodes),
        "candidate_policy": {
            "name": "verified_candidate_filtering",
            "enable_fallback": True,
            "always_include_fallback": False,
            "description": (
                "Generate proposals, keep the candidates that pass hard "
                "verification, select the verified candidate with minimum "
                "exact latency, and invoke fallback only when the verified "
                "candidate set is empty."
            ),
        },
        "budget_definition": {
            "symbol": "B_NN",
            "sequential_gnn": (
                "For a target B_NN, the number of stochastic trajectories is "
                "max(1, floor(B_NN / number_of_services)).  The evidence also "
                "reports the realized average budget."
            ),
        },
        "methods": {
            f"sequential_b{budget}": {
                "family": "sequential",
                "model": "sequential_conditional_gnn",
                "target_forward_equivalent_budget": int(budget),
                "samples": "max(1, floor(B_NN / num_services))",
                "stochastic": True,
            }
            for budget in budgets
        },
    }


def _selected_indices(dataset: LabeledDeploymentDataset, limit: int | None) -> list[int]:
    indices = list(range(len(dataset)))
    return indices if limit is None else indices[:limit]


def _inference_config(protocol: Mapping[str, Any], samples: int) -> InferenceConfig:
    return InferenceConfig(
        num_samples=samples,
        sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
        fallback_max_search_nodes=int(protocol["fallback_max_search_nodes"]),
        enable_repair=False,
        enable_fallback=True,
        always_include_fallback=False,
    )


def _sequential_samples(instance: Any, budget: int) -> int:
    return max(1, int(budget) // max(1, int(instance.num_services)))


def _payload(result: Any, pool_best: float, sampled: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result.metrics
    raw = metrics.get("best_raw_objective")
    return {
        "success": bool(result.success),
        "source": result.source,
        "objective": result.objective,
        "gap_to_pool_best": (
            None if result.objective is None else float(result.objective) / pool_best - 1.0
        ),
        "raw_success": raw is not None,
        "raw_gap_to_pool_best": None if raw is None else float(raw) / pool_best - 1.0,
        "raw_any_feasible": bool(metrics.get("raw_any_feasible", False)),
        "raw_feasible_count": int(metrics.get("raw_feasible_count", 0)),
        "num_raw_proposals": int(metrics.get("num_raw_proposals", 0)),
        "raw_feasible_rate": metrics.get("raw_feasible_rate"),
        "raw_unique_rate": metrics.get("raw_unique_rate"),
        "raw_pairwise_hamming": metrics.get("raw_pairwise_hamming"),
        "fallback_invoked": bool(metrics.get("fallback_invoked", False)),
        "fallback_success": bool(metrics.get("fallback_success", False)),
        "sampling_seconds": float(metrics.get("sampling_seconds", 0.0)),
        "verification_seconds": float(metrics.get("verification_seconds", 0.0)),
        "fallback_seconds": float(metrics.get("fallback_seconds", 0.0)),
        "total_seconds": float(metrics.get("total_seconds", 0.0)),
        "completed_rate": float(sampled["completed_rate"]),
        "samples": int(sampled["samples"]),
        "realized_budget": int(sampled["realized_budget"]),
        "sequential_model_forwards": int(sampled["sequential_model_forwards"]),
    }


def _record_valid(
    path: Path,
    *,
    protocol_hash: str,
    checkpoint_sha256: str,
    dataset_freeze_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    row = _read_json(path)
    return (
        row.get("scope") == RECORD_SCOPE
        and row.get("protocol_sha256") == protocol_hash
        and row.get("sequential_checkpoint_sha256") == checkpoint_sha256
        and row.get("dataset_freeze_sha256") == dataset_freeze_sha256
    )


def run_evaluation(
    root: Path,
    *,
    selected_seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    checkpoint_root: Path,
    output_root: Path,
    device_name: str,
    max_instances: int | None,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    protocol = _protocol(
        root=root,
        selected_seeds=selected_seeds,
        budgets=budgets,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        device=device_name,
        max_instances=max_instances,
        fallback_max_search_nodes=fallback_max_search_nodes,
    )
    protocol_hash = _hash_payload(protocol)
    seed_everything(int(protocol["evaluation_seed"]), deterministic=True)

    dataset_root = _resolve(root, DATASET_ROOT)
    freeze = audit_dataset_freeze(dataset_root)
    dataset_hash = file_sha256(dataset_root / "dataset_freeze.json")
    dataset = LabeledDeploymentDataset(dataset_root, partitions=(PARTITION,), require_freeze=True)
    indices = _selected_indices(dataset, max_instances)
    device = _device(device_name)

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "protocol.json", protocol)

    completed = 0
    for seed in selected_seeds:
        checkpoint = _sequential_checkpoint(checkpoint_root, int(seed))
        checkpoint_hash = file_sha256(checkpoint)
        solver = _SEQ.load_sequential_solver(checkpoint, dataset, device)

        for index in indices:
            item = dataset[index]
            record_path = (
                output_root
                / "records"
                / str(seed)
                / item.partition
                / f"{item.instance.instance_id}.json"
            )
            if _record_valid(
                record_path,
                protocol_hash=protocol_hash,
                checkpoint_sha256=checkpoint_hash,
                dataset_freeze_sha256=dataset_hash,
            ):
                completed += 1
                continue

            pool_best = float(np.min(item.pool.latencies))
            method_results: dict[str, Any] = {}
            for budget in budgets:
                method_id = f"sequential_b{budget}"
                samples = _sequential_samples(item.instance, int(budget))
                generator = torch.Generator(device=device).manual_seed(
                    derive_seed(
                        int(protocol["evaluation_seed"]),
                        (
                            f"{SCOPE}:{method_id}:{seed}:"
                            f"{item.partition}:{item.instance.instance_id}"
                        ),
                    )
                )
                sampled_batch = sample_sequential_proposals(
                    solver.model,
                    item.instance,
                    solver.feature_schema,
                    config=SequentialDecodeConfig(
                        num_samples=samples,
                        sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
                        stochastic=True,
                        temperature=float(protocol["temperature"]),
                    ),
                    device=device,
                    generator=generator,
                )
                sampled = {
                    "proposals": sampled_batch.proposals,
                    "probabilities": sampled_batch.probabilities,
                    "seconds": sampled_batch.sampling_seconds,
                    "completed_rate": float(sampled_batch.completed.mean()),
                    "samples": samples,
                    "realized_budget": samples * int(item.instance.num_services),
                    "sequential_model_forwards": sampled_batch.model_forwards,
                }
                result = solve_from_proposals(
                    item.instance,
                    sampled["proposals"],
                    model_probabilities=sampled["probabilities"],
                    config=_inference_config(protocol, samples),
                    sampling_seconds=float(sampled["seconds"]),
                    proposal_method=method_id,
                )
                method_results[method_id] = _payload(result, pool_best, sampled)

            write_json(
                record_path,
                {
                    "schema_version": "1.0",
                    "scope": RECORD_SCOPE,
                    "partition": item.partition,
                    "seed": int(seed),
                    "instance_id": item.instance.instance_id,
                    "num_services": int(item.instance.num_services),
                    "num_devices": int(item.instance.num_devices),
                    "num_dependencies": int(item.instance.dependency_index.shape[1]),
                    "pool_best": pool_best,
                    "pool_size": int(item.pool.size),
                    "dataset_family": freeze.get("dataset_name"),
                    "protocol_sha256": protocol_hash,
                    "sequential_checkpoint": _relative(root, checkpoint),
                    "sequential_checkpoint_sha256": checkpoint_hash,
                    "dataset_freeze_sha256": dataset_hash,
                    "methods": method_results,
                },
            )
            completed += 1

    return {
        "scope": SCOPE,
        "records_completed": completed,
        "expected": len(indices) * len(selected_seeds),
        "budgets": list(budgets),
        "output_root": _relative(root, output_root),
    }


def _records(output_root: Path) -> list[Mapping[str, Any]]:
    return [_read_json(path) for path in sorted((output_root / "records").rglob("*.json"))]


def _finite(values: Sequence[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _mean(values: Sequence[float | None]) -> float | None:
    data = _finite(values)
    return mean(data) if data else None


def _std(values: Sequence[float | None]) -> float | None:
    data = _finite(values)
    if not data:
        return None
    return pstdev(data) if len(data) > 1 else 0.0


def _aggregate(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_proposals = sum(int(payload["num_raw_proposals"]) for payload in payloads)
    total_feasible = sum(int(payload["raw_feasible_count"]) for payload in payloads)
    return {
        "records": len(payloads),
        "success_rate": mean(float(payload["success"]) for payload in payloads),
        "mean_gap_to_pool_best": _mean([payload["gap_to_pool_best"] for payload in payloads]),
        "gap_std": _std([payload["gap_to_pool_best"] for payload in payloads]),
        "verified_proposal_rate": mean(float(payload["raw_success"]) for payload in payloads),
        "proposal_feasible_rate": total_feasible / total_proposals if total_proposals else None,
        "fallback_invocation_rate": mean(float(payload["fallback_invoked"]) for payload in payloads),
        "mean_total_seconds": _mean([payload["total_seconds"] for payload in payloads]),
        "mean_sampling_seconds": _mean([payload["sampling_seconds"] for payload in payloads]),
        "mean_realized_budget": _mean([payload["realized_budget"] for payload in payloads]),
    }


def _sign_test_p_value(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    if n <= 1024:
        return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))
    mean_n = n / 2.0
    sigma = math.sqrt(n / 4.0)
    z = (k + 0.5 - mean_n) / sigma
    return min(1.0, 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0)))


def _direct_masked_records(records_root: Path) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    rows = [_read_json(path) for path in sorted(records_root.rglob("*.json"))]
    return {
        (int(row["training_seed"]), str(row["partition"]), str(row["instance_id"])): row
        for row in rows
    }


def _paired(
    rows: Sequence[Mapping[str, Any]],
    phase_index: Mapping[tuple[int, str, str], Mapping[str, Any]],
    budget: int,
    right_method: str,
) -> dict[str, Any]:
    wins = losses = ties = skipped = missing = 0
    left_id = f"sequential_b{budget}"
    right_id = f"{right_method}_b{budget}"
    for row in rows:
        key = (int(row["seed"]), str(row["partition"]), str(row["instance_id"]))
        phase = phase_index.get(key)
        if phase is None:
            missing += 1
            continue
        left = row["methods"][left_id]["gap_to_pool_best"]
        right = phase["methods"][right_id]["gap_to_pool_best"]
        if left is None or right is None:
            skipped += 1
            continue
        if float(left) < float(right) - 1e-12:
            wins += 1
        elif float(right) < float(left) - 1e-12:
            losses += 1
        else:
            ties += 1
    return {
        "budget": int(budget),
        "left": left_id,
        "right": right_id,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "skipped": skipped,
        "missing_phase6f_records": missing,
        "p_value_two_sided_sign_test": _sign_test_p_value(wins, losses),
    }


def finalize(
    root: Path,
    *,
    output_root: Path,
    direct_masked_records_root: Path,
) -> dict[str, Any]:
    rows = _records(output_root)
    if not rows:
        raise RuntimeError(f"No records found under {output_root / 'records'}")
    protocol = _read_json(output_root / "protocol.json")
    budgets = [int(value) for value in protocol["budgets"]]
    phase_index = _direct_masked_records(direct_masked_records_root)

    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "matched_direct_masked_records": len(phase_index),
        "output_root": _relative(root, output_root),
        "budgets": budgets,
        "methods": protocol["methods"],
        "overall": {},
        "paired": {},
    }
    for budget in budgets:
        method_id = f"sequential_b{budget}"
        evidence["overall"][method_id] = _aggregate([row["methods"][method_id] for row in rows])
        evidence["paired"][f"sequential_vs_direct_b{budget}"] = _paired(
            rows,
            phase_index,
            budget,
            "direct",
        )
        evidence["paired"][f"sequential_vs_masked_b{budget}"] = _paired(
            rows,
            phase_index,
            budget,
            "masked",
        )

    evidence_path = output_root / "sequential_forward_budget_evidence.json"
    report_path = output_root / "sequential_forward_budget_report.md"
    write_json(evidence_path, evidence)
    report_path.write_text(_report(evidence), encoding="utf-8")
    return {
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "evidence": _relative(root, evidence_path),
        "report": _relative(root, report_path),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _report(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 6F Sequential Forward-Budget Calibration",
        "",
        f"Records: {evidence['records']}",
        "",
        "| B_NN | Samples policy | Verified proposal | Proposal feasible | Final gap | Fallback | Realized B_NN | Time |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    methods = evidence["methods"]
    for budget in evidence["budgets"]:
        method_id = f"sequential_b{budget}"
        row = evidence["overall"][method_id]
        lines.append(
            f"| {budget} | {methods[method_id]['samples']} | "
            f"{_pct(row['verified_proposal_rate'])} | "
            f"{_pct(row['proposal_feasible_rate'])} | "
            f"{_pct(row['mean_gap_to_pool_best'])} | "
            f"{_pct(row['fallback_invocation_rate'])} | "
            f"{_num(row['mean_realized_budget'])} | "
            f"{_num(row['mean_total_seconds'])} s |"
        )
    lines.extend(["", "## Paired Final-Gap Tests", ""])
    for budget in evidence["budgets"]:
        for right in ("direct", "masked"):
            row = evidence["paired"][f"sequential_vs_{right}_b{budget}"]
            lines.append(
                f"- B_NN={budget}, sequential/{right}/tie/skipped/missing="
                f"{row['wins']}/{row['losses']}/{row['ties']}/"
                f"{row['skipped']}/{row['missing_phase6f_records']}, "
                f"p={_num(row['p_value_two_sided_sign_test'], 6)}."
            )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "finalize", "all"))
    parser.add_argument(
        "--checkpoint-root",
        default="artifacts/phase6f-sequential-conditional-training",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase6f-sequential-forward-budget-calibration",
    )
    parser.add_argument(
        "--direct-masked-records-root",
        default="artifacts/phase6f-forward-budget-calibration/records",
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fallback-max-search-nodes", type=int, default=30000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = _resolve(root, args.output_root)
    budgets = _parse_budgets(args.budgets)
    if args.action in {"run", "all"}:
        print(
            run_evaluation(
                root,
                selected_seeds=_parse_seeds(args.seeds),
                budgets=budgets,
                checkpoint_root=_resolve(root, args.checkpoint_root),
                output_root=output_root,
                device_name=args.device,
                max_instances=args.max_instances,
                fallback_max_search_nodes=int(args.fallback_max_search_nodes),
            ),
            flush=True,
        )
    if args.action in {"finalize", "all"}:
        print(
            finalize(
                root,
                output_root=output_root,
                direct_masked_records_root=_resolve(root, args.direct_masked_records_root),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
