"""Forward-equivalent budget calibration for Phase 6F neural proposals.

This script sweeps the neural forward-equivalent budget on the sealed test set
for the two core neural proposal generators: one-shot Direct prediction and
absorbing-MASK diffusion.  Raw proposals are filtered by the hard verifier, the
best verified candidate is selected by exact end-to-end latency, and the
constructive fallback is invoked only if a method produces no verified
candidate for an instance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.experiments.phase6ee_stage3 import load_stage3_solver
from gdm_factor_diffusion.experiments.schema import file_sha256
from gdm_factor_diffusion.inference import InferenceConfig, solve_from_proposals
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze


SCOPE = "phase6f_forward_budget_calibration"
RECORD_SCOPE = f"{SCOPE}_record"
EVIDENCE_SCOPE = f"{SCOPE}_evidence"

DEFAULT_SEEDS = tuple(range(2026070111, 2026070121))
DEFAULT_BUDGETS = (8, 16, 32, 64, 128)
MASKED_STEPS_PER_PROPOSAL = 8

DATASET_ROOT = "artifacts/datasets/phase6e-e-stage38-sealed"
PARTITION = "sealed_test_id"


def _load_stage39_helpers() -> Any:
    path = Path(__file__).with_name("74_run_phase6e_e_stage39_forward_budget.py")
    spec = importlib.util.spec_from_file_location("phase6e_stage39_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Stage 3.9 helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STAGE39 = _load_stage39_helpers()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(root: Path, path: str | Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parse_int_csv(value: str | None, *, default: Sequence[int]) -> tuple[int, ...]:
    if value is None or not value.strip():
        return tuple(default)
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_seeds(value: str | None) -> tuple[int, ...]:
    seeds = _parse_int_csv(value, default=DEFAULT_SEEDS)
    unknown = sorted(set(seeds) - set(DEFAULT_SEEDS))
    if unknown:
        raise ValueError(f"Unsupported ten-seed freeze seeds: {unknown}")
    return seeds


def _parse_budgets(value: str | None) -> tuple[int, ...]:
    budgets = _parse_int_csv(value, default=DEFAULT_BUDGETS)
    invalid = [budget for budget in budgets if budget < MASKED_STEPS_PER_PROPOSAL]
    if invalid:
        raise ValueError(f"Budgets must be >= {MASKED_STEPS_PER_PROPOSAL}: {invalid}")
    non_multiple = [budget for budget in budgets if budget % MASKED_STEPS_PER_PROPOSAL]
    if non_multiple:
        raise ValueError(
            "Budgets must be multiples of the masked reverse-step count "
            f"({MASKED_STEPS_PER_PROPOSAL}): {non_multiple}"
        )
    return tuple(sorted(dict.fromkeys(budgets)))


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _load_training_entry(
    root: Path,
    training: Mapping[str, Any],
    seed: int,
    kind: str,
) -> Path:
    return _resolve(root, training["runs"][str(seed)][kind]["paths"]["best.pt"])


def _method_specs(budgets: Sequence[int]) -> dict[str, dict[str, Any]]:
    methods: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        methods[f"direct_b{budget}"] = {
            "family": "direct",
            "model": "direct",
            "samples": int(budget),
            "neural_steps_per_proposal": 1,
            "forward_equivalent_budget": int(budget),
        }
        methods[f"masked_b{budget}"] = {
            "family": "masked",
            "model": "masked_diffusion",
            "samples": int(budget // MASKED_STEPS_PER_PROPOSAL),
            "stochastic": True,
            "neural_steps_per_proposal": MASKED_STEPS_PER_PROPOSAL,
            "forward_equivalent_budget": int(budget),
        }
    return methods


def _protocol(
    *,
    root: Path,
    selected_seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    training_freeze: Path,
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
        "training_freeze": _relative(root, training_freeze),
        "output_root": _relative(root, output_root),
        "device": device,
        "deterministic": True,
        "evaluation_seed": 2026071211,
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
            "direct": "B_NN equals the number of sampled one-shot proposals.",
            "masked_diffusion": (
                "B_NN equals the number of sampled proposals multiplied by "
                f"{MASKED_STEPS_PER_PROPOSAL} reverse denoising steps."
            ),
        },
        "methods": _method_specs(budgets),
    }


def _inference_config(protocol: Mapping[str, Any], samples: int) -> InferenceConfig:
    return InferenceConfig(
        num_samples=samples,
        sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
        fallback_max_search_nodes=int(protocol["fallback_max_search_nodes"]),
        enable_repair=False,
        enable_fallback=True,
        always_include_fallback=False,
    )


def _payload(result: Any, pool_best: float, sampled: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result.metrics
    raw = metrics.get("best_raw_objective")
    payload = {
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
        "realized_budget": sampled.get("realized_budget"),
    }
    for key in ("masked_model_forwards", "masked_completed_rate"):
        if key in sampled:
            payload[key] = sampled[key]
    return payload


def _record_valid(
    path: Path,
    *,
    protocol_hash: str,
    training_freeze_sha256: str,
    dataset_freeze_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    row = _read_json(path)
    return (
        row.get("scope") == RECORD_SCOPE
        and row.get("protocol_sha256") == protocol_hash
        and row.get("training_freeze_sha256") == training_freeze_sha256
        and row.get("dataset_freeze_sha256") == dataset_freeze_sha256
    )


def _selected_indices(dataset: LabeledDeploymentDataset, limit: int | None) -> list[int]:
    indices = list(range(len(dataset)))
    return indices if limit is None else indices[:limit]


def run_evaluation(
    root: Path,
    *,
    selected_seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    training_freeze: Path,
    output_root: Path,
    device_name: str,
    max_instances: int | None,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    protocol = _protocol(
        root=root,
        selected_seeds=selected_seeds,
        budgets=budgets,
        training_freeze=training_freeze,
        output_root=output_root,
        device=device_name,
        max_instances=max_instances,
        fallback_max_search_nodes=fallback_max_search_nodes,
    )
    protocol_hash = _STAGE39._hash_payload(protocol)
    seed_everything(int(protocol["evaluation_seed"]), deterministic=True)

    training = _read_json(training_freeze)
    if training.get("scope") != "phase6e_e_stage39_forward_budget_training_freeze":
        raise ValueError("Expected the Stage 3.9 ten-seed training freeze.")

    dataset_root = _resolve(root, DATASET_ROOT)
    freeze = audit_dataset_freeze(dataset_root)
    dataset_hash = file_sha256(dataset_root / "dataset_freeze.json")
    training_hash = file_sha256(training_freeze)
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=(PARTITION,),
        require_freeze=True,
    )
    indices = _selected_indices(dataset, max_instances)
    device = _device(device_name)

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "protocol.json", protocol)

    completed = 0
    for training_seed in selected_seeds:
        direct = load_stage3_solver(
            _load_training_entry(root, training, int(training_seed), "direct"),
            dataset,
            device,
        )
        masked = load_stage3_solver(
            _load_training_entry(root, training, int(training_seed), "masked_conditional"),
            dataset,
            device,
        )
        if masked.schedule is None:
            raise ValueError("Masked checkpoint has no absorbing-MASK schedule.")

        for index in indices:
            item = dataset[index]
            record_path = (
                output_root
                / "records"
                / str(training_seed)
                / item.partition
                / f"{item.instance.instance_id}.json"
            )
            if _record_valid(
                record_path,
                protocol_hash=protocol_hash,
                training_freeze_sha256=training_hash,
                dataset_freeze_sha256=dataset_hash,
            ):
                completed += 1
                continue

            pool_best = float(np.min(item.pool.latencies))
            method_results: dict[str, Any] = {}
            for method_id, method in protocol["methods"].items():
                generator = torch.Generator(device=device).manual_seed(
                    derive_seed(
                        int(protocol["evaluation_seed"]),
                        (
                            f"{SCOPE}:{method_id}:{training_seed}:"
                            f"{item.partition}:{item.instance.instance_id}"
                        ),
                    )
                )
                sampled = _STAGE39._sample_proposals(
                    method_id=method_id,
                    method=method,
                    direct_solver=direct,
                    masked_solver=masked,
                    instance=item.instance,
                    protocol=protocol,
                    device=device,
                    generator=generator,
                )
                sampled["realized_budget"] = (
                    int(method["samples"]) * int(method["neural_steps_per_proposal"])
                )
                result = solve_from_proposals(
                    item.instance,
                    sampled["proposals"],
                    model_probabilities=sampled["probabilities"],
                    config=_inference_config(protocol, int(method["samples"])),
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
                    "training_seed": int(training_seed),
                    "instance_id": item.instance.instance_id,
                    "num_services": int(item.instance.num_services),
                    "num_devices": int(item.instance.num_devices),
                    "num_dependencies": int(item.instance.dependency_index.shape[1]),
                    "pool_best": pool_best,
                    "pool_size": int(item.pool.size),
                    "dataset_family": freeze.get("dataset_name"),
                    "protocol_sha256": protocol_hash,
                    "training_freeze_sha256": training_hash,
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


def _finite_values(values: Sequence[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _finite_mean(values: Sequence[float | None]) -> float | None:
    finite = _finite_values(values)
    return mean(finite) if finite else None


def _finite_std(values: Sequence[float | None]) -> float | None:
    finite = _finite_values(values)
    if not finite:
        return None
    return pstdev(finite) if len(finite) > 1 else 0.0


def _aggregate(rows: list[Mapping[str, Any]], method_id: str) -> dict[str, Any]:
    payloads = [row["methods"][method_id] for row in rows]
    total_proposals = sum(int(payload["num_raw_proposals"]) for payload in payloads)
    total_feasible = sum(int(payload["raw_feasible_count"]) for payload in payloads)
    source_counts = {source: 0 for source in ("raw", "fallback", "failure")}
    for payload in payloads:
        source = str(payload["source"])
        source_counts[source if source in source_counts else "failure"] += 1
    records = len(payloads)
    return {
        "records": records,
        "success_rate": mean(float(payload["success"]) for payload in payloads),
        "mean_gap_to_pool_best": _finite_mean(
            [payload["gap_to_pool_best"] for payload in payloads]
        ),
        "gap_std": _finite_std([payload["gap_to_pool_best"] for payload in payloads]),
        "verified_proposal_rate": mean(float(payload["raw_success"]) for payload in payloads),
        "proposal_feasible_rate": total_feasible / total_proposals if total_proposals else None,
        "fallback_invocation_rate": mean(
            float(payload["fallback_invoked"]) for payload in payloads
        ),
        "source_rates": {
            source: count / records if records else None for source, count in source_counts.items()
        },
        "mean_total_seconds": _finite_mean([payload["total_seconds"] for payload in payloads]),
        "mean_sampling_seconds": _finite_mean([payload["sampling_seconds"] for payload in payloads]),
        "mean_realized_budget": _finite_mean([payload["realized_budget"] for payload in payloads]),
    }


def _sign_test_p_value(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    if n <= 1024:
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        return min(1.0, 2.0 * tail)
    mean_n = n / 2.0
    sigma = math.sqrt(n / 4.0)
    z = (k + 0.5 - mean_n) / sigma
    return min(1.0, 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0)))


def _paired(rows: list[Mapping[str, Any]], budget: int) -> dict[str, Any]:
    direct_id = f"direct_b{budget}"
    masked_id = f"masked_b{budget}"
    wins = losses = ties = skipped = 0
    for row in rows:
        masked = row["methods"][masked_id]["gap_to_pool_best"]
        direct = row["methods"][direct_id]["gap_to_pool_best"]
        if masked is None or direct is None:
            skipped += 1
            continue
        if float(masked) < float(direct) - 1e-12:
            wins += 1
        elif float(direct) < float(masked) - 1e-12:
            losses += 1
        else:
            ties += 1
    return {
        "budget": int(budget),
        "left": masked_id,
        "right": direct_id,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "skipped": skipped,
        "p_value_two_sided_sign_test": _sign_test_p_value(wins, losses),
    }


def finalize(root: Path, *, output_root: Path) -> dict[str, Any]:
    rows = _records(output_root)
    if not rows:
        raise RuntimeError(f"No records found under {output_root / 'records'}")
    protocol = _read_json(output_root / "protocol.json")
    budgets = [int(value) for value in protocol["budgets"]]

    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "output_root": _relative(root, output_root),
        "budgets": budgets,
        "methods": protocol["methods"],
        "overall": {},
        "paired": {},
    }
    for budget in budgets:
        evidence["overall"][f"direct_b{budget}"] = _aggregate(rows, f"direct_b{budget}")
        evidence["overall"][f"masked_b{budget}"] = _aggregate(rows, f"masked_b{budget}")
        evidence["paired"][str(budget)] = _paired(rows, budget)

    evidence_path = output_root / "forward_budget_calibration_evidence.json"
    report_path = output_root / "forward_budget_calibration_report.md"
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
    return "n/a" if value is None else f"{value:.{digits}f}"


def _report(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 6F Forward-Budget Calibration",
        "",
        f"Records: {evidence['records']}",
        "",
        "| B_NN | Method | Samples | Success | Verified proposal | Proposal feasible | Final gap | Fallback | Time |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    methods = evidence["methods"]
    for budget in evidence["budgets"]:
        for method_id in (f"direct_b{budget}", f"masked_b{budget}"):
            row = evidence["overall"][method_id]
            method = methods[method_id]
            name = "Direct" if method["model"] == "direct" else "Masked Diffusion"
            lines.append(
                f"| {budget} | {name} | {method['samples']} | "
                f"{_pct(row['success_rate'])} | "
                f"{_pct(row['verified_proposal_rate'])} | "
                f"{_pct(row['proposal_feasible_rate'])} | "
                f"{_pct(row['mean_gap_to_pool_best'])} | "
                f"{_pct(row['fallback_invocation_rate'])} | "
                f"{_num(row['mean_total_seconds'])} s |"
            )
    lines.extend(["", "## Paired Final-Gap Tests", ""])
    for budget in evidence["budgets"]:
        row = evidence["paired"][str(budget)]
        lines.append(
            f"- B_NN={budget}: masked/direct/tie/skipped="
            f"{row['wins']}/{row['losses']}/{row['ties']}/{row['skipped']}, "
            f"p={_num(row['p_value_two_sided_sign_test'], 6)}."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "finalize", "all"))
    parser.add_argument(
        "--training-freeze",
        default=(
            "artifacts/phase6e-e-stage39-10seed-training/"
            "ten_seed_training_freeze.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase6f-forward-budget-calibration",
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
    selected_seeds = _parse_seeds(args.seeds)
    budgets = _parse_budgets(args.budgets)
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("--max-instances must be positive.")
    kwargs = {
        "selected_seeds": selected_seeds,
        "budgets": budgets,
        "training_freeze": _resolve(root, args.training_freeze),
        "output_root": _resolve(root, args.output_root),
        "device_name": args.device,
        "max_instances": args.max_instances,
        "fallback_max_search_nodes": int(args.fallback_max_search_nodes),
    }
    if args.action in {"run", "all"}:
        print(run_evaluation(root, **kwargs), flush=True)
    if args.action in {"finalize", "all"}:
        print(finalize(root, output_root=kwargs["output_root"]), flush=True)


if __name__ == "__main__":
    main()
