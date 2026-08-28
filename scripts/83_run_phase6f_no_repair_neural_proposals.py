"""Phase 6F no-repair neural proposal evaluation.

This evaluation isolates neural proposal quality. Each learned method generates
raw deployment proposals, the hard verifier filters feasible candidates, and
the exact latency objective selects the best verified raw candidate. The
deterministic fallback is invoked only when no verified raw candidate exists.
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
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    solve_fallback_only,
    solve_from_proposals,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze


SCOPE = "phase6f_no_repair_neural_proposals"
RECORD_SCOPE = f"{SCOPE}_seed_instance_record"
EVIDENCE_SCOPE = f"{SCOPE}_evidence"

DEFAULT_SEEDS = tuple(range(2026070111, 2026070121))
MAIN_DIRECT = "direct_k64"
MAIN_MASKED = "masked_diffusion_k8"
METHOD_ORDER = (
    "fallback_only",
    "random_k64",
    MAIN_DIRECT,
    "masked_deterministic_k1",
    MAIN_MASKED,
)

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "sealed_id": {
        "dataset_root": "artifacts/datasets/phase6e-e-stage38-sealed",
        "partitions": ("sealed_test_id",),
    },
    "controlled_shift": {
        "dataset_root": "artifacts/datasets/phase6e-e-controlled-shift",
        "partitions": (
            "test_id_reference",
            "shift_high_sharing",
            "shift_low_compatibility",
            "shift_tight_capacity",
            "shift_unseen_workflow",
        ),
    },
    "realistic_profile": {
        "dataset_root": "artifacts/datasets/phase6e-e-realistic-profile",
        "partitions": ("profile_id", "profile_branched", "profile_high_sharing"),
    },
}


def _stage39_module() -> Any:
    path = Path(__file__).with_name("74_run_phase6e_e_stage39_forward_budget.py")
    spec = importlib.util.spec_from_file_location("phase6e_stage39_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Stage 3.9 helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STAGE39 = _stage39_module()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(root: Path, path: str | Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parse_csv(value: str | None, *, default: Sequence[str]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return tuple(default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_seed_list(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return DEFAULT_SEEDS
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    unknown = sorted(set(seeds) - set(DEFAULT_SEEDS))
    if unknown:
        raise ValueError(f"Unsupported Stage 3.9 seeds: {unknown}")
    return seeds


def _parse_setting_list(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return tuple(DATASET_SPECS)
    settings = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(settings) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(f"Unsupported settings: {unknown}")
    return settings


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _limited_indices(
    dataset: LabeledDeploymentDataset,
    max_per_partition: int | None,
) -> list[int]:
    selected: list[int] = []
    seen: dict[str, int] = defaultdict(int)
    for index, (_, pool_entry) in enumerate(dataset.entries):
        partition = str(pool_entry["partition"])
        if max_per_partition is not None and seen[partition] >= max_per_partition:
            continue
        selected.append(index)
        seen[partition] += 1
    return selected


def _protocol(
    *,
    root: Path,
    settings: tuple[str, ...],
    selected_seeds: tuple[int, ...],
    training_freeze: Path,
    output_root: Path,
    device: str,
    max_instances_per_partition: int | None,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scope": SCOPE,
        "settings": list(settings),
        "selected_seeds": list(selected_seeds),
        "training_freeze": _relative(root, training_freeze),
        "output_root": _relative(root, output_root),
        "device": device,
        "deterministic": True,
        "evaluation_seed": 2026071011,
        "sample_batch_size": 8,
        "temperature": 1.0,
        "max_instances_per_partition": max_instances_per_partition,
        "fallback_max_search_nodes": int(fallback_max_search_nodes),
        "policy": {
            "name": "verified_raw_then_fallback",
            "enable_repair": False,
            "enable_fallback": True,
            "always_include_fallback": False,
            "description": (
                "Generate raw proposals, filter them using the hard verifier, "
                "select the minimum exact-latency verified raw candidate, and "
                "invoke fallback only when no verified raw candidate exists."
            ),
        },
        "methods": {
            "fallback_only": {
                "family": "fallback",
                "samples": 0,
                "neural_steps_per_proposal": 0,
                "forward_equivalent_budget": 0,
            },
            "random_k64": {
                "family": "random",
                "samples": 64,
                "neural_steps_per_proposal": 0,
                "forward_equivalent_budget": 0,
            },
            MAIN_DIRECT: {
                "family": "direct",
                "samples": 64,
                "neural_steps_per_proposal": 1,
                "forward_equivalent_budget": 64,
            },
            "masked_deterministic_k1": {
                "family": "masked",
                "samples": 1,
                "stochastic": False,
                "neural_steps_per_proposal": 8,
                "forward_equivalent_budget": 8,
            },
            MAIN_MASKED: {
                "family": "masked",
                "samples": 8,
                "stochastic": True,
                "neural_steps_per_proposal": 8,
                "forward_equivalent_budget": 64,
            },
        },
        "main_comparison": {
            "direct": MAIN_DIRECT,
            "masked": MAIN_MASKED,
            "budget": 64,
            "budget_definition": "B_NN = N_prop * N_step",
        },
        "dataset_specs": {
            name: {
                "dataset_root": spec["dataset_root"],
                "partitions": list(spec["partitions"]),
            }
            for name, spec in DATASET_SPECS.items()
            if name in settings
        },
    }


def _inference_config(protocol: Mapping[str, Any], samples: int) -> InferenceConfig:
    policy = protocol["policy"]
    return InferenceConfig(
        num_samples=samples,
        sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
        fallback_max_search_nodes=int(protocol["fallback_max_search_nodes"]),
        enable_repair=bool(policy["enable_repair"]),
        enable_fallback=bool(policy["enable_fallback"]),
        always_include_fallback=bool(policy["always_include_fallback"]),
    )


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


def _payload(result: Any, pool_best: float) -> dict[str, Any]:
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
        "raw_gap_to_pool_best": (
            None if raw is None else float(raw) / pool_best - 1.0
        ),
        "raw_any_feasible": bool(metrics.get("raw_any_feasible", False)),
        "raw_feasible_count": int(metrics.get("raw_feasible_count", 0)),
        "num_raw_proposals": int(metrics.get("num_raw_proposals", 0)),
        "raw_feasible_rate": metrics.get("raw_feasible_rate"),
        "raw_unique_rate": metrics.get("raw_unique_rate"),
        "raw_pairwise_hamming": metrics.get("raw_pairwise_hamming"),
        "raw_capacity_violation_rate": metrics.get("raw_capacity_violation_rate"),
        "raw_link_violation_rate": metrics.get("raw_link_violation_rate"),
        "fallback_invoked": bool(metrics.get("fallback_invoked", False)),
        "fallback_success": bool(metrics.get("fallback_success", False)),
        "fallback_seconds": float(metrics.get("fallback_seconds", 0.0)),
        "sampling_seconds": float(metrics.get("sampling_seconds", 0.0)),
        "verification_seconds": float(metrics.get("verification_seconds", 0.0)),
        "exact_evaluation_seconds": float(metrics.get("exact_evaluation_seconds", 0.0)),
        "selection_seconds": float(metrics.get("selection_seconds", 0.0)),
        "total_seconds": float(metrics.get("total_seconds", 0.0)),
        "masked_model_forwards": metrics.get("masked_model_forwards"),
        "masked_completed_rate": metrics.get("masked_completed_rate"),
    }


def _load_training_entry(
    root: Path,
    training: Mapping[str, Any],
    seed: int,
    kind: str,
) -> Path:
    return _resolve(root, training["runs"][str(seed)][kind]["paths"]["best.pt"])


def _run_one_setting(
    root: Path,
    *,
    setting: str,
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_hash: str,
    training: Mapping[str, Any],
    training_freeze: Path,
    output_root: Path,
) -> dict[str, Any]:
    dataset_root = _resolve(root, spec["dataset_root"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root for {setting}: {dataset_root}")
    freeze = audit_dataset_freeze(dataset_root)
    dataset_hash = file_sha256(dataset_root / "dataset_freeze.json")
    training_hash = file_sha256(training_freeze)
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=tuple(spec["partitions"]),
        require_freeze=True,
    )
    indices = _limited_indices(dataset, protocol["max_instances_per_partition"])
    device = _device(str(protocol["device"]))
    methods = protocol["methods"]
    completed = 0

    for training_seed in protocol["selected_seeds"]:
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
                / setting
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
            for method_id, method in methods.items():
                if method["family"] == "fallback":
                    result = solve_fallback_only(
                        item.instance,
                        max_search_nodes=int(protocol["fallback_max_search_nodes"]),
                    )
                    method_results[method_id] = {
                        "verify_fallback": _payload(result, pool_best)
                    }
                    continue
                sampled = _STAGE39._sample_proposals(
                    method_id=method_id,
                    method=method,
                    direct_solver=direct,
                    masked_solver=masked,
                    instance=item.instance,
                    protocol=protocol,
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(
                        derive_seed(
                            int(protocol["evaluation_seed"]),
                            (
                                f"phase6f:{setting}:{method_id}:{training_seed}:"
                                f"{item.partition}:{item.instance.instance_id}"
                            ),
                        )
                    ),
                )
                result = solve_from_proposals(
                    item.instance,
                    sampled["proposals"],
                    model_probabilities=sampled["probabilities"],
                    config=_inference_config(protocol, int(method["samples"])),
                    sampling_seconds=float(sampled["seconds"]),
                    proposal_method=method_id,
                )
                payload = _payload(result, pool_best)
                if "masked_model_forwards" in sampled:
                    payload["masked_model_forwards"] = int(sampled["masked_model_forwards"])
                    payload["masked_completed_rate"] = float(sampled["masked_completed_rate"])
                method_results[method_id] = {"verify_fallback": payload}

            write_json(
                record_path,
                {
                    "schema_version": "1.0",
                    "scope": RECORD_SCOPE,
                    "setting": setting,
                    "training_seed": int(training_seed),
                    "instance_id": item.instance.instance_id,
                    "partition": item.partition,
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
        "setting": setting,
        "records_completed": completed,
        "expected": len(indices) * len(protocol["selected_seeds"]),
        "partitions": list(spec["partitions"]),
        "dataset_root": _relative(root, dataset_root),
    }


def run_evaluation(
    root: Path,
    *,
    settings: tuple[str, ...],
    selected_seeds: tuple[int, ...],
    training_freeze: Path,
    output_root: Path,
    device_name: str,
    max_instances_per_partition: int | None,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    protocol = _protocol(
        root=root,
        settings=settings,
        selected_seeds=selected_seeds,
        training_freeze=training_freeze,
        output_root=output_root,
        device=device_name,
        max_instances_per_partition=max_instances_per_partition,
        fallback_max_search_nodes=fallback_max_search_nodes,
    )
    protocol_hash = _STAGE39._hash_payload(protocol)
    seed_everything(int(protocol["evaluation_seed"]), deterministic=True)
    training = _read_json(training_freeze)
    if training.get("scope") != "phase6e_e_stage39_forward_budget_training_freeze":
        raise ValueError("Expected the Stage 3.9 ten-seed training freeze.")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "protocol.json", protocol)
    summaries = []
    for setting in settings:
        summaries.append(
            _run_one_setting(
                root,
                setting=setting,
                spec=DATASET_SPECS[setting],
                protocol=protocol,
                protocol_hash=protocol_hash,
                training=training,
                training_freeze=training_freeze,
                output_root=output_root,
            )
        )
    return {
        "scope": SCOPE,
        "protocol_sha256": protocol_hash,
        "settings": list(settings),
        "selected_seeds": list(selected_seeds),
        "output_root": _relative(root, output_root),
        "summaries": summaries,
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
    payloads = [row["methods"][method_id]["verify_fallback"] for row in rows]
    source_counts = {source: 0 for source in ("raw", "fallback", "failure")}
    for payload in payloads:
        source = str(payload["source"])
        source_counts[source if source in source_counts else "failure"] += 1
    total_proposals = sum(int(payload["num_raw_proposals"]) for payload in payloads)
    total_feasible = sum(int(payload["raw_feasible_count"]) for payload in payloads)
    records = len(payloads)
    return {
        "records": records,
        "success_rate": mean(float(payload["success"]) for payload in payloads),
        "mean_gap_to_pool_best": _finite_mean(
            [payload["gap_to_pool_best"] for payload in payloads]
        ),
        "gap_std": _finite_std([payload["gap_to_pool_best"] for payload in payloads]),
        "raw_any_feasible_rate": mean(float(payload["raw_success"]) for payload in payloads),
        "proposal_feasible_rate": (
            total_feasible / total_proposals if total_proposals else None
        ),
        "infeasible_proposal_rate": (
            1.0 - total_feasible / total_proposals if total_proposals else None
        ),
        "mean_raw_gap_to_pool_best": _finite_mean(
            [payload["raw_gap_to_pool_best"] for payload in payloads]
        ),
        "raw_gap_std": _finite_std([payload["raw_gap_to_pool_best"] for payload in payloads]),
        "mean_raw_unique_rate": _finite_mean([payload["raw_unique_rate"] for payload in payloads]),
        "mean_raw_pairwise_hamming": _finite_mean(
            [payload["raw_pairwise_hamming"] for payload in payloads]
        ),
        "fallback_invocation_rate": mean(
            float(payload["fallback_invoked"]) for payload in payloads
        ),
        "source_rates": {
            source: count / records if records else None
            for source, count in source_counts.items()
        },
        "mean_total_seconds": _finite_mean([payload["total_seconds"] for payload in payloads]),
        "mean_sampling_seconds": _finite_mean(
            [payload["sampling_seconds"] for payload in payloads]
        ),
        "mean_verification_seconds": _finite_mean(
            [payload["verification_seconds"] for payload in payloads]
        ),
        "mean_fallback_seconds": _finite_mean(
            [payload["fallback_seconds"] for payload in payloads]
        ),
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


def _paired(rows: list[Mapping[str, Any]], *, stage: str) -> dict[str, Any]:
    wins = losses = ties = skipped = 0
    for row in rows:
        direct = row["methods"][MAIN_DIRECT]["verify_fallback"]
        masked = row["methods"][MAIN_MASKED]["verify_fallback"]
        key = "raw_gap_to_pool_best" if stage == "raw" else "gap_to_pool_best"
        left = masked[key]
        right = direct[key]
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
        "stage": stage,
        "left": MAIN_MASKED,
        "right": MAIN_DIRECT,
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
    protocol_path = output_root / "protocol.json"
    protocol = _read_json(protocol_path) if protocol_path.is_file() else {}

    by_setting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setting[str(row["setting"])].append(row)

    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "output_root": _relative(root, output_root),
        "methods": protocol.get("methods", {}),
        "overall": {},
        "by_setting": {},
        "paired": {},
    }
    for method_id in (MAIN_DIRECT, MAIN_MASKED):
        evidence["overall"][method_id] = _aggregate(rows, method_id)
    for method_id in METHOD_ORDER:
        if method_id not in evidence["overall"]:
            evidence["overall"][method_id] = _aggregate(rows, method_id)
    evidence["paired"]["raw"] = _paired(rows, stage="raw")
    evidence["paired"]["final"] = _paired(rows, stage="final")

    for setting, setting_rows in sorted(by_setting.items()):
        evidence["by_setting"][setting] = {
            method_id: _aggregate(setting_rows, method_id)
            for method_id in METHOD_ORDER
        }
        evidence["paired"][setting] = {
            "raw": _paired(setting_rows, stage="raw"),
            "final": _paired(setting_rows, stage="final"),
        }

    evidence_path = output_root / "no_repair_neural_proposal_evidence.json"
    report_path = output_root / "no_repair_neural_proposal_report.md"
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


def _method_row(
    method_id: str,
    row: Mapping[str, Any],
    method_meta: Mapping[str, Any] | None = None,
) -> str:
    source = row["source_rates"]
    budget = "n/a" if method_meta is None else str(method_meta.get("forward_equivalent_budget", "n/a"))
    return (
        f"| {method_id} | {budget} | {_pct(row['success_rate'])} | "
        f"{_pct(row['proposal_feasible_rate'])} | "
        f"{_pct(row['raw_any_feasible_rate'])} | "
        f"{_pct(row['mean_raw_gap_to_pool_best'])} | "
        f"{_pct(row['mean_gap_to_pool_best'])} | "
        f"{_pct(row['fallback_invocation_rate'])} | "
        f"{_pct(source['raw'])}/{_pct(source['fallback'])}/{_pct(source['failure'])} | "
        f"{_num(row['mean_total_seconds'])} s |"
    )


def _paired_line(name: str, row: Mapping[str, Any]) -> str:
    p = row["p_value_two_sided_sign_test"]
    return (
        f"- {name}: wins/losses/ties/skipped="
        f"{row['wins']}/{row['losses']}/{row['ties']}/{row['skipped']}, "
        f"p={_num(p, 6)}."
    )


def _report(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 6F No-Repair Neural Proposal Evaluation",
        "",
        "Policy: raw proposals are filtered by the hard verifier. If at least one",
        "verified raw candidate exists, the solver selects the exact minimum-latency",
        "raw candidate. Fallback is invoked only when no verified raw candidate exists.",
        "",
        f"Records: {evidence['records']}",
        "",
        "## Overall",
        "",
        "| Method | B_NN | Success | Proposal feasible | Any feasible | Raw gap | Final gap | Fallback invoked | Source(raw/fallback/failure) | Time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    method_meta = evidence.get("methods", {})
    for method_id in METHOD_ORDER:
        lines.append(
            _method_row(method_id, evidence["overall"][method_id], method_meta.get(method_id))
        )
    lines.extend(
        [
            "",
            "## Paired Tests",
            "",
            _paired_line("raw", evidence["paired"]["raw"]),
            _paired_line("final", evidence["paired"]["final"]),
            "",
            "## By Setting",
            "",
        ]
    )
    for setting, methods in evidence["by_setting"].items():
        lines.extend(
            [
                f"### {setting}",
                "",
                "| Method | B_NN | Success | Proposal feasible | Any feasible | Raw gap | Final gap | Fallback invoked | Source(raw/fallback/failure) | Time |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method_id in METHOD_ORDER:
            lines.append(_method_row(method_id, methods[method_id], method_meta.get(method_id)))
        lines.extend(
            [
                "",
                _paired_line("raw", evidence["paired"][setting]["raw"]),
                _paired_line("final", evidence["paired"][setting]["final"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "- These results isolate neural proposal quality under a verified raw-candidate selection policy.",
            "- Fallback protects final feasibility only when all raw proposals fail verification.",
            "- The reported gap is relative to the best verified MILP-pool member, not a proven global optimum unless separately certified.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "finalize", "all"))
    parser.add_argument("--settings", default=",".join(DATASET_SPECS))
    parser.add_argument(
        "--training-freeze",
        default=(
            "artifacts/phase6e-e-stage39-10seed-training/"
            "ten_seed_training_freeze.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase6f-no-repair-neural-proposals",
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--max-instances-per-partition", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fallback-max-search-nodes", type=int, default=30000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = _parse_setting_list(args.settings)
    selected_seeds = _parse_seed_list(args.seeds)
    if args.max_instances_per_partition is not None and args.max_instances_per_partition < 1:
        raise ValueError("--max-instances-per-partition must be positive.")
    kwargs = {
        "settings": settings,
        "selected_seeds": selected_seeds,
        "training_freeze": _resolve(root, args.training_freeze),
        "output_root": _resolve(root, args.output_root),
        "device_name": args.device,
        "max_instances_per_partition": args.max_instances_per_partition,
        "fallback_max_search_nodes": int(args.fallback_max_search_nodes),
    }
    if args.action in {"run", "all"}:
        print(run_evaluation(root, **kwargs), flush=True)
    if args.action in {"finalize", "all"}:
        print(finalize(root, output_root=kwargs["output_root"]), flush=True)


if __name__ == "__main__":
    main()
