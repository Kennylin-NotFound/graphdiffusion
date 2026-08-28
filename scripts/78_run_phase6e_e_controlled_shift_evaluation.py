"""Controlled-shift robustness evaluation for the absorbing-MASK solver.

The script evaluates frozen Stage 3.9 checkpoints on test partitions that
modify one distribution factor at a time. It reuses the same proposal and
post-processing protocol as the cross-scale evaluation, while adding stronger
non-learned heuristic baselines.
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
    solve_greedy_local,
    solve_latency_aware_heuristic,
    solve_local_search,
    solve_from_proposals,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze


SCOPE = "phase6e_e_controlled_shift"
RECORD_SCOPE = f"{SCOPE}_seed_instance_record"
EVIDENCE_SCOPE = f"{SCOPE}_evidence"

DEFAULT_SEEDS = tuple(range(2026070111, 2026070121))
DEFAULT_PARTITIONS = (
    "test_id_reference",
    "shift_high_sharing",
    "shift_low_compatibility",
    "shift_tight_capacity",
    "shift_unseen_workflow",
)
MAIN_DIRECT = "direct_k64"
MAIN_MASKED = "masked_diffusion_k8"
METHOD_PROFILES = ("full", "core", "lean")


def _cross_module() -> Any:
    path = Path(__file__).with_name("76_run_phase6e_e_cross_scale_evaluation.py")
    spec = importlib.util.spec_from_file_location("phase6e_cross_scale_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load cross-scale helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CROSS = _cross_module()


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


def _validate_method_profile(value: str) -> str:
    if value not in METHOD_PROFILES:
        raise ValueError(
            f"Unsupported method profile {value!r}; expected one of {METHOD_PROFILES}."
        )
    return value


def _protocol(
    *,
    root: Path,
    dataset_root: Path,
    training_freeze: Path,
    output_root: Path,
    partitions: tuple[str, ...],
    max_instances_per_partition: int | None,
    device: str,
    method_profile: str,
    repair_candidate_limit: int | None,
    repair_max_moves: int,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    methods = {
        "fallback_only": {
            "family": "fallback",
            "samples": 0,
            "neural_steps_per_proposal": 0,
            "forward_equivalent_budget": 0,
        },
        "greedy_local": {
            "family": "greedy",
            "samples": 1,
            "neural_steps_per_proposal": 0,
            "forward_equivalent_budget": 0,
        },
        "latency_aware_heuristic": {
            "family": "heuristic",
            "samples": 6,
            "neural_steps_per_proposal": 0,
            "forward_equivalent_budget": 0,
        },
        "local_search": {
            "family": "local_search",
            "samples": 1,
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
    }
    if method_profile == "core":
        methods.pop("fallback_only")
        methods.pop("random_k64")
    elif method_profile == "lean":
        methods = {
            key: value
            for key, value in methods.items()
            if key in {
                "latency_aware_heuristic",
                "local_search",
                MAIN_DIRECT,
                "masked_deterministic_k1",
                MAIN_MASKED,
            }
        }

    return {
        "schema_version": "1.0",
        "scope": SCOPE,
        "dataset_root": _relative(root, dataset_root),
        "training_freeze": _relative(root, training_freeze),
        "output_root": _relative(root, output_root),
        "partitions": list(partitions),
        "max_instances_per_partition": max_instances_per_partition,
        "device": device,
        "method_profile": method_profile,
        "deterministic": True,
        "evaluation_seed": 2026070811,
        "sample_batch_size": 8,
        "temperature": 1.0,
        "repair_max_moves": int(repair_max_moves),
        "repair_candidate_limit": repair_candidate_limit,
        "fallback_max_search_nodes": int(fallback_max_search_nodes),
        "main_comparison": {
            "direct": MAIN_DIRECT,
            "masked": MAIN_MASKED,
            "budget": 64,
            "budget_definition": "B_NN = N_prop * N_step",
        },
        "methods": methods,
        "postprocessing_modes": {
            "raw_only": {
                "enable_repair": False,
                "enable_fallback": False,
                "always_include_fallback": False,
            },
            "repair_only": {
                "enable_repair": True,
                "enable_fallback": False,
                "always_include_fallback": False,
            },
            "full": {
                "enable_repair": True,
                "enable_fallback": True,
                "always_include_fallback": True,
            },
        },
        "claim_boundary": (
            "Controlled-shift robustness is measured under synthetic, "
            "single-factor distribution shifts. It is not a real-trace claim."
        ),
    }


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


def _inference_config(
    protocol: Mapping[str, Any],
    samples: int,
    mode: str,
) -> InferenceConfig:
    post = protocol["postprocessing_modes"][mode]
    return InferenceConfig(
        num_samples=max(1, samples),
        sample_batch_size=min(int(protocol["sample_batch_size"]), max(1, samples)),
        repair_max_moves=int(protocol["repair_max_moves"]),
        fallback_max_search_nodes=int(protocol["fallback_max_search_nodes"]),
        enable_repair=bool(post["enable_repair"]),
        enable_fallback=bool(post["enable_fallback"]),
        always_include_fallback=bool(post["always_include_fallback"]),
        repair_candidate_limit=protocol.get("repair_candidate_limit"),
    )


def _load_training_entry(
    root: Path,
    training: Mapping[str, Any],
    seed: int,
    kind: str,
) -> Path:
    return _resolve(root, training["runs"][str(seed)][kind]["paths"]["best.pt"])


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


def _run_baseline(method_id: str, instance: Any, protocol: Mapping[str, Any]) -> Any:
    if method_id == "fallback_only":
        return solve_fallback_only(
            instance,
            max_search_nodes=int(protocol["fallback_max_search_nodes"]),
        )
    if method_id == "greedy_local":
        return solve_greedy_local(instance)
    if method_id == "latency_aware_heuristic":
        return solve_latency_aware_heuristic(instance)
    if method_id == "local_search":
        return solve_local_search(instance)
    raise ValueError(f"Unsupported deterministic baseline: {method_id}")


def run_evaluation(
    root: Path,
    *,
    dataset_root: Path,
    training_freeze: Path,
    output_root: Path,
    selected_seeds: tuple[int, ...],
    partitions: tuple[str, ...],
    max_instances_per_partition: int | None,
    device_name: str,
    method_profile: str,
    repair_candidate_limit: int | None,
    repair_max_moves: int,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    protocol = _protocol(
        root=root,
        dataset_root=dataset_root,
        training_freeze=training_freeze,
        output_root=output_root,
        partitions=partitions,
        max_instances_per_partition=max_instances_per_partition,
        device=device_name,
        method_profile=method_profile,
        repair_candidate_limit=repair_candidate_limit,
        repair_max_moves=repair_max_moves,
        fallback_max_search_nodes=fallback_max_search_nodes,
    )
    protocol_hash = _CROSS._hash_payload(protocol)
    device = _CROSS._device(device_name)
    seed_everything(
        int(protocol["evaluation_seed"]),
        deterministic=bool(protocol["deterministic"]),
    )

    freeze = audit_dataset_freeze(dataset_root)
    dataset_hash = file_sha256(dataset_root / "dataset_freeze.json")
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=partitions,
        require_freeze=True,
    )
    indices = _limited_indices(dataset, max_instances_per_partition)
    training = _read_json(training_freeze)
    if training.get("scope") != "phase6e_e_stage39_forward_budget_training_freeze":
        raise ValueError("Expected the Stage 3.9 ten-seed training freeze.")
    training_hash = file_sha256(training_freeze)
    records_root = output_root / "records"
    methods = protocol["methods"]
    modes = protocol["postprocessing_modes"]
    learned_methods = [
        method_id
        for method_id, method in methods.items()
        if method["family"] in {"direct", "masked", "random"}
    ]
    baseline_methods = [
        method_id
        for method_id, method in methods.items()
        if method["family"] in {"fallback", "greedy", "heuristic", "local_search"}
    ]
    completed = 0

    for training_seed in selected_seeds:
        direct = load_stage3_solver(
            _load_training_entry(root, training, training_seed, "direct"),
            dataset,
            device,
        )
        masked = load_stage3_solver(
            _load_training_entry(root, training, training_seed, "masked_conditional"),
            dataset,
            device,
        )
        if masked.schedule is None:
            raise ValueError("Masked checkpoint has no absorbing-MASK schedule.")

        for index in indices:
            item = dataset[index]
            record_path = (
                records_root
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
            for method_id in learned_methods:
                method = methods[method_id]
                sampled = _CROSS._sample_proposals(
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
                                f"controlled-shift:{method_id}:{training_seed}:"
                                f"{item.partition}:{item.instance.instance_id}"
                            ),
                        )
                    ),
                )
                mode_payload: dict[str, Any] = {}
                for mode in modes:
                    result = solve_from_proposals(
                        item.instance,
                        sampled["proposals"],
                        model_probabilities=sampled["probabilities"],
                        config=_inference_config(protocol, int(method["samples"]), mode),
                        sampling_seconds=float(sampled["seconds"]),
                        proposal_method=method_id,
                    )
                    payload = _CROSS._result_payload(result, pool_best)
                    if "masked_model_forwards" in sampled:
                        payload["masked_model_forwards"] = int(
                            sampled["masked_model_forwards"]
                        )
                        payload["masked_completed_rate"] = float(
                            sampled["masked_completed_rate"]
                        )
                    mode_payload[mode] = payload
                method_results[method_id] = mode_payload

            for method_id in baseline_methods:
                result = _run_baseline(method_id, item.instance, protocol)
                method_results[method_id] = {
                    "full": _CROSS._result_payload(result, pool_best)
                }

            write_json(
                record_path,
                {
                    "schema_version": "1.0",
                    "scope": RECORD_SCOPE,
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
        "records_completed": completed,
        "expected": len(indices) * len(selected_seeds),
        "selected_seeds": list(selected_seeds),
        "partitions": list(partitions),
        "max_instances_per_partition": max_instances_per_partition,
        "output_root": _relative(root, output_root),
        "device": str(device),
    }


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return mean(finite) if finite else None


def _finite_std(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return pstdev(finite) if len(finite) > 1 else 0.0


def _aggregate(records: list[Mapping[str, Any]], method_id: str, mode: str) -> dict[str, Any]:
    rows = [
        row["methods"][method_id][mode]
        for row in records
        if mode in row["methods"][method_id]
    ]
    sources = (
        "raw",
        "repair",
        "fallback",
        "greedy_local",
        "latency_aware_heuristic",
        "local_search",
        "failure",
    )
    return {
        "records": len(rows),
        "success_rate": mean(float(row["success"]) for row in rows),
        "mean_gap_to_pool_best": _finite_mean(
            [row["gap_to_pool_best"] for row in rows]
        ),
        "gap_std": _finite_std([row["gap_to_pool_best"] for row in rows]),
        "raw_success_rate": mean(float(row["raw_success"]) for row in rows),
        "mean_raw_gap_to_pool_best": _finite_mean(
            [row["raw_gap_to_pool_best"] for row in rows]
        ),
        "pre_fallback_success_rate": mean(
            float(row["pre_fallback_success"]) for row in rows
        ),
        "mean_pre_fallback_gap": _finite_mean(
            [row["pre_fallback_gap"] for row in rows]
        ),
        "raw_any_feasibility": mean(float(row["raw_any_feasible"]) for row in rows),
        "mean_raw_feasible_rate": _finite_mean(
            [row["raw_feasible_rate"] for row in rows]
        ),
        "mean_raw_unique_rate": _finite_mean([row["raw_unique_rate"] for row in rows]),
        "mean_raw_pairwise_hamming": _finite_mean(
            [row["raw_pairwise_hamming"] for row in rows]
        ),
        "mean_capacity_violation_rate": _finite_mean(
            [row["raw_capacity_violation_rate"] for row in rows]
        ),
        "mean_link_violation_rate": _finite_mean(
            [row["raw_link_violation_rate"] for row in rows]
        ),
        "repair_attempts_mean": mean(float(row["repair_attempts"]) for row in rows),
        "repair_success_rate_mean": mean(
            float(row["repair_success_rate"]) for row in rows
        ),
        "fallback_invocation_rate": mean(float(row["fallback_invoked"]) for row in rows),
        "mean_total_seconds": mean(float(row["total_seconds"]) for row in rows),
        "source_rates": {
            source: mean(float(row["source"] == source) for row in rows)
            for source in sources
        },
    }


def _score(rows: list[Mapping[str, Any]], stage: str) -> tuple[float, float]:
    if stage == "raw":
        success = mean(float(row["raw_success"]) for row in rows)
        gaps = [
            float(row["raw_gap_to_pool_best"])
            for row in rows
            if row["raw_success"] and row["raw_gap_to_pool_best"] is not None
        ]
    elif stage == "pre":
        success = mean(float(row["pre_fallback_success"]) for row in rows)
        gaps = [
            float(row["pre_fallback_gap"])
            for row in rows
            if row["pre_fallback_success"] and row["pre_fallback_gap"] is not None
        ]
    elif stage == "final":
        success = mean(float(row["success"]) for row in rows)
        gaps = [
            float(row["gap_to_pool_best"])
            for row in rows
            if row["success"] and row["gap_to_pool_best"] is not None
        ]
    else:
        raise ValueError(f"Unsupported paired stage: {stage}")
    return (-success, mean(gaps) if gaps else float("inf"))


def _sign_test_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _paired(
    records: list[Mapping[str, Any]],
    *,
    direct: str,
    masked: str,
    mode: str,
    stage: str,
) -> dict[str, Any]:
    by_instance: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_instance[f"{row['partition']}:{row['instance_id']}"].append(row)
    wins = losses = ties = 0
    for rows in by_instance.values():
        direct_rows = [row["methods"][direct][mode] for row in rows]
        masked_rows = [row["methods"][masked][mode] for row in rows]
        direct_score = _score(direct_rows, stage)
        masked_score = _score(masked_rows, stage)
        if masked_score < direct_score:
            wins += 1
        elif direct_score < masked_score:
            losses += 1
        else:
            ties += 1
    return {
        "mode": mode,
        "stage": stage,
        "masked_wins": wins,
        "direct_wins": losses,
        "ties": ties,
        "sign_test_pvalue": _sign_test_pvalue(wins, losses),
    }


def _load_records(records_root: Path) -> list[dict[str, Any]]:
    return [_read_json(path) for path in sorted(records_root.glob("*/*/*.json"))]


def _write_report(path: Path, evidence: Mapping[str, Any]) -> None:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * float(value):.3f}%"

    lines = [
        "# Phase 6E-E Controlled-Shift Evaluation",
        "",
        "This report evaluates frozen Stage 3.9 checkpoints under synthetic",
        "single-factor distribution shifts. It should be used for robustness",
        "discussion, not as real-trace evidence.",
        "",
        "## Dataset",
        "",
        f"- Dataset root: `{evidence['protocol']['dataset_root']}`",
        f"- Partitions: `{', '.join(evidence['protocol']['partitions'])}`",
        f"- Max instances per partition: `{evidence['protocol']['max_instances_per_partition']}`",
        f"- Records: `{evidence['records']}`",
        "",
        "## Partition Summary",
        "",
        "| Partition | Method | Raw succ. | Raw gap | Pre gap | Final gap | Fallback selected | Time |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    method_order = [
        "fallback_only",
        "greedy_local",
        "latency_aware_heuristic",
        "local_search",
        "random_k64",
        MAIN_DIRECT,
        "masked_deterministic_k1",
        MAIN_MASKED,
    ]
    available_methods = set(evidence["protocol"]["methods"])
    for partition, methods in evidence["by_partition"].items():
        for method_id in method_order:
            if method_id not in available_methods:
                continue
            full = methods[method_id]["full"]
            raw = methods[method_id].get("raw_only", full)
            lines.append(
                f"| {partition} | {method_id} | "
                f"{100 * raw['raw_success_rate']:.2f}% | "
                f"{pct(raw['mean_raw_gap_to_pool_best'])} | "
                f"{pct(full['mean_pre_fallback_gap'])} | "
                f"{pct(full['mean_gap_to_pool_best'])} | "
                f"{100 * full['source_rates']['fallback']:.2f}% | "
                f"{full['mean_total_seconds']:.3f} s |"
            )
    lines.extend(
        [
            "",
            "## Masked-vs-Direct Paired Tests By Partition",
            "",
            "| Partition | Stage | Masked wins | Direct wins | Ties | p-value |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for partition, stages in evidence["paired_by_partition"].items():
        for stage, result in stages.items():
            lines.append(
                f"| {partition} | {stage} | {result['masked_wins']} | "
                f"{result['direct_wins']} | {result['ties']} | "
                f"{result['sign_test_pvalue']:.6g} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(
    root: Path,
    *,
    dataset_root: Path,
    training_freeze: Path,
    output_root: Path,
    partitions: tuple[str, ...],
    max_instances_per_partition: int | None,
    device_name: str,
    selected_seeds: tuple[int, ...],
    method_profile: str,
    repair_candidate_limit: int | None,
    repair_max_moves: int,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    protocol = _protocol(
        root=root,
        dataset_root=dataset_root,
        training_freeze=training_freeze,
        output_root=output_root,
        partitions=partitions,
        max_instances_per_partition=max_instances_per_partition,
        device=device_name,
        method_profile=method_profile,
        repair_candidate_limit=repair_candidate_limit,
        repair_max_moves=repair_max_moves,
        fallback_max_search_nodes=fallback_max_search_nodes,
    )
    protocol_hash = _CROSS._hash_payload(protocol)
    dataset_hash = file_sha256(dataset_root / "dataset_freeze.json")
    training_hash = file_sha256(training_freeze)
    records = _load_records(output_root / "records")
    valid = [
        row
        for row in records
        if row.get("scope") == RECORD_SCOPE
        and row.get("protocol_sha256") == protocol_hash
        and row.get("training_freeze_sha256") == training_hash
        and row.get("dataset_freeze_sha256") == dataset_hash
        and int(row.get("training_seed")) in set(selected_seeds)
    ]
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=partitions,
        require_freeze=True,
    )
    expected_instances = len(_limited_indices(dataset, max_instances_per_partition))
    expected = expected_instances * len(selected_seeds)
    if len(valid) != expected:
        raise ValueError(f"Expected {expected} records, found {len(valid)}.")

    methods = protocol["methods"]
    modes = protocol["postprocessing_modes"]
    method_modes = {
        method_id: (
            ("full",)
            if method["family"] in {"fallback", "greedy", "heuristic", "local_search"}
            else tuple(modes.keys())
        )
        for method_id, method in methods.items()
    }
    overall = {
        method_id: {
            mode: _aggregate(valid, method_id, mode)
            for mode in method_modes[method_id]
        }
        for method_id in methods
    }
    by_partition = {}
    paired_by_partition = {}
    for partition in partitions:
        rows = [row for row in valid if row["partition"] == partition]
        by_partition[partition] = {
            method_id: {
                mode: _aggregate(rows, method_id, mode)
                for mode in method_modes[method_id]
            }
            for method_id in methods
        }
        if MAIN_DIRECT in methods and MAIN_MASKED in methods:
            paired_by_partition[partition] = {
                "raw": _paired(
                    rows,
                    direct=MAIN_DIRECT,
                    masked=MAIN_MASKED,
                    mode="raw_only",
                    stage="raw",
                ),
                "pre": _paired(
                    rows,
                    direct=MAIN_DIRECT,
                    masked=MAIN_MASKED,
                    mode="full",
                    stage="pre",
                ),
                "final": _paired(
                    rows,
                    direct=MAIN_DIRECT,
                    masked=MAIN_MASKED,
                    mode="full",
                    stage="final",
                ),
            }
        else:
            paired_by_partition[partition] = {}

    evidence = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "training_freeze": _relative(root, training_freeze),
        "training_freeze_sha256": training_hash,
        "dataset_freeze_sha256": dataset_hash,
        "records": len(valid),
        "instances": expected_instances,
        "seeds": list(selected_seeds),
        "overall": overall,
        "by_partition": by_partition,
        "paired_by_partition": paired_by_partition,
        "claim_boundary": protocol["claim_boundary"],
    }
    evidence_path = output_root / "controlled_shift_evidence.json"
    report_path = output_root / "controlled_shift_report.md"
    write_json(evidence_path, evidence)
    _write_report(report_path, evidence)
    return {
        "evidence": _relative(root, evidence_path),
        "evidence_sha256": file_sha256(evidence_path),
        "report": _relative(root, report_path),
        "records": len(valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "finalize", "all"))
    parser.add_argument(
        "--dataset-root",
        default="artifacts/datasets/phase6e-e-controlled-shift",
    )
    parser.add_argument(
        "--training-freeze",
        default=(
            "artifacts/phase6e-e-stage39-10seed-training/"
            "ten_seed_training_freeze.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase6e-e-controlled-shift-evaluation",
    )
    parser.add_argument(
        "--partitions",
        default=",".join(DEFAULT_PARTITIONS),
        help="Comma-separated controlled-shift partitions.",
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-instances-per-partition", type=int, default=None)
    parser.add_argument(
        "--method-profile",
        choices=METHOD_PROFILES,
        default="full",
        help=(
            "full keeps all diagnostic baselines; core drops fallback-only and "
            "random; lean keeps heuristic/local-search plus Direct/Masked variants."
        ),
    )
    parser.add_argument("--repair-candidate-limit", type=int, default=None)
    parser.add_argument("--repair-max-moves", type=int, default=10)
    parser.add_argument("--fallback-max-search-nodes", type=int, default=100_000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_root = _resolve(root, args.dataset_root)
    training_freeze = _resolve(root, args.training_freeze)
    output_root = _resolve(root, args.output_root)
    partitions = _parse_csv(args.partitions, default=DEFAULT_PARTITIONS)
    selected_seeds = _parse_seed_list(args.seeds)
    method_profile = _validate_method_profile(args.method_profile)
    if args.max_instances_per_partition is not None and args.max_instances_per_partition < 1:
        raise ValueError("--max-instances-per-partition must be positive.")
    if args.repair_candidate_limit is not None and args.repair_candidate_limit < 1:
        raise ValueError("--repair-candidate-limit must be positive.")
    if args.repair_max_moves < 0:
        raise ValueError("--repair-max-moves must be nonnegative.")
    if args.fallback_max_search_nodes < 1:
        raise ValueError("--fallback-max-search-nodes must be positive.")

    kwargs = {
        "dataset_root": dataset_root,
        "training_freeze": training_freeze,
        "output_root": output_root,
        "partitions": partitions,
        "max_instances_per_partition": args.max_instances_per_partition,
        "device_name": args.device,
        "selected_seeds": selected_seeds,
        "method_profile": method_profile,
        "repair_candidate_limit": args.repair_candidate_limit,
        "repair_max_moves": args.repair_max_moves,
        "fallback_max_search_nodes": args.fallback_max_search_nodes,
    }
    if args.action in {"run", "all"}:
        print(run_evaluation(root, **kwargs))
    if args.action in {"finalize", "all"}:
        print(finalize(root, **kwargs))


if __name__ == "__main__":
    main()
