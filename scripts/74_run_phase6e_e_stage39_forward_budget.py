"""Ten-seed forward-equivalent evaluation for the absorbing-MASK solver.

This script does not retrain models.  It combines the frozen Stage 3.8 seeds
with the Stage 3.9 cloud-extension seeds, evaluates them on the sealed ID
dataset, and reports the main comparison under the hardware-stable neural
forward-equivalent budget:

    Direct K=64, one neural pass per proposal  -> B_NN = 64
    Masked diffusion K=8, eight completion steps -> B_NN = 64

The output is intentionally separate from Stage 3.8 so that no frozen evidence
is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.experiments.phase6ee_stage3 import load_stage3_solver
from gdm_factor_diffusion.experiments.schema import file_sha256
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    sample_direct_proposals,
    sample_masked_proposals,
    sample_random_proposals,
    solve_fallback_only,
    solve_from_proposals,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze


SCOPE = "phase6e_e_stage39_forward_budget"
TRAINING_FREEZE_SCOPE = f"{SCOPE}_training_freeze"
RECORD_SCOPE = f"{SCOPE}_seed_instance_record"
EVIDENCE_SCOPE = f"{SCOPE}_evidence"

SEEDS = tuple(range(2026070111, 2026070121))
STAGE38_SEEDS = (2026070111, 2026070112, 2026070113)
STAGE39_SEEDS = tuple(range(2026070114, 2026070121))

DIRECT_K_VALUES = (8, 16, 32, 64, 128)
MASKED_K_VALUES = (2, 4, 8, 16)
MAIN_DIRECT = "direct_k64"
MAIN_MASKED = "masked_diffusion_k8"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _relative(root: Path, path: str | Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _rank(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
    gap = record["mean_pre_fallback_gap"]
    return (
        -float(record["pre_fallback_success_rate"]),
        float("inf") if gap is None else float(gap),
        -float(record["raw_any_feasibility"]),
        float(record["mean_online_seconds"]),
    )


def _hash_payload(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest().upper()


def _stage39_summary(root: Path, run_directory: Path) -> dict[str, Any]:
    paths = {
        name: run_directory / name
        for name in ("config.json", "metrics.jsonl", "best.pt", "latest.pt")
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_directory}: {missing}")
    records = _read_jsonl(paths["metrics.jsonl"])
    train = [row for row in records if row.get("type") == "train"]
    selection = [row for row in records if row.get("type") == "checkpoint_selection"]
    if not train or not selection:
        raise ValueError(f"Missing metrics records in {run_directory}.")
    best_record = min(selection, key=_rank)
    best = torch.load(paths["best.pt"], map_location="cpu", weights_only=False)
    latest = torch.load(paths["latest.pt"], map_location="cpu", weights_only=False)
    if int(best["step"]) != int(best_record["step"]):
        raise ValueError(f"Best checkpoint mismatch in {run_directory}.")
    final_step = int(train[-1]["step"])
    if int(latest["step"]) != final_step:
        raise ValueError(f"Latest checkpoint mismatch in {run_directory}.")
    return {
        "run": run_directory.name,
        "final_step": final_step,
        "best_step": int(best_record["step"]),
        "best_selection": {
            key: best_record[key]
            for key in (
                "final_verified_rate",
                "pre_fallback_success_rate",
                "mean_pre_fallback_gap",
                "raw_any_feasibility",
                "mean_online_seconds",
            )
        },
        "paths": {name: _relative(root, path) for name, path in paths.items()},
        "sha256": {name: file_sha256(path) for name, path in paths.items()},
    }


def freeze_training(root: Path) -> dict[str, Any]:
    """Create a ten-seed training-freeze file from existing checkpoints."""

    stage38_path = root / "artifacts" / "phase6e-e-stage38-training" / "training_freeze.json"
    stage38 = _read_json(stage38_path)
    if stage38.get("scope") != "phase6e_e_stage38_multiseed_training_freeze":
        raise ValueError("Unexpected Stage 3.8 training freeze scope.")

    runs: dict[str, Any] = {}
    for seed in STAGE38_SEEDS:
        runs[str(seed)] = stage38["runs"][str(seed)]
        for kind in ("direct", "masked_conditional"):
            for path in runs[str(seed)][kind]["paths"].values():
                resolved = _resolve(root, path)
                if not resolved.is_file():
                    raise FileNotFoundError(f"Missing Stage 3.8 checkpoint artifact: {resolved}")

    stage39_root = root / "artifacts" / "phase6e-e-stage39-10seed-training"
    for seed in STAGE39_SEEDS:
        runs[str(seed)] = {
            kind: _stage39_summary(root, stage39_root / f"{kind}-seed{seed}")
            for kind in ("direct", "masked_conditional")
        }

    freeze = {
        "schema_version": "1.0",
        "scope": TRAINING_FREEZE_SCOPE,
        "source": {
            "stage38_training_freeze": _relative(root, stage38_path),
            "stage38_training_freeze_sha256": file_sha256(stage38_path),
            "stage39_training_root": _relative(root, stage39_root),
        },
        "training_seeds": list(SEEDS),
        "runs": runs,
        "sealed_data_opened": True,
        "note": (
            "This freeze combines already frozen Stage 3.8 seeds with the Stage "
            "3.9 cloud-extension seeds. It is for ten-seed evaluation only."
        ),
    }
    output = stage39_root / "ten_seed_training_freeze.json"
    if output.exists() and _read_json(output) != freeze:
        raise ValueError("Existing ten-seed training freeze disagrees with current runs.")
    write_json(output, freeze)
    return {"training_freeze": _relative(root, output), "sha256": file_sha256(output)}


def _evaluation_protocol() -> dict[str, Any]:
    methods: dict[str, Any] = {
        f"direct_k{k}": {
            "family": "direct",
            "samples": k,
            "neural_steps_per_proposal": 1,
            "forward_equivalent_budget": k,
        }
        for k in DIRECT_K_VALUES
    }
    methods["masked_deterministic_k1"] = {
        "family": "masked",
        "samples": 1,
        "stochastic": False,
        "neural_steps_per_proposal": 8,
        "forward_equivalent_budget": 8,
    }
    for k in MASKED_K_VALUES:
        methods[f"masked_diffusion_k{k}"] = {
            "family": "masked",
            "samples": k,
            "stochastic": True,
            "neural_steps_per_proposal": 8,
            "forward_equivalent_budget": 8 * k,
        }
    methods["random_k64"] = {
        "family": "random",
        "samples": 64,
        "neural_steps_per_proposal": 0,
        "forward_equivalent_budget": 0,
    }
    methods["fallback_only"] = {
        "family": "fallback",
        "samples": 0,
        "neural_steps_per_proposal": 0,
        "forward_equivalent_budget": 0,
    }
    return {
        "schema_version": "1.0",
        "scope": SCOPE,
        "dataset_root": "artifacts/datasets/phase6e-e-stage38-sealed",
        "partition": "sealed_test_id",
        "expected_instances": 128,
        "training_freeze": (
            "artifacts/phase6e-e-stage39-10seed-training/"
            "ten_seed_training_freeze.json"
        ),
        "output_root": "artifacts/phase6e-e-stage39-forward-budget-evaluation",
        "device": "cuda",
        "deterministic": True,
        "evaluation_seed": 2026070611,
        "sample_batch_size": 8,
        "temperature": 1.0,
        "repair_max_moves": 10,
        "fallback_max_search_nodes": 100_000,
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
    }


def _inference_config(protocol: Mapping[str, Any], samples: int, mode: str) -> InferenceConfig:
    post = protocol["postprocessing_modes"][mode]
    return InferenceConfig(
        num_samples=samples,
        sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
        repair_max_moves=int(protocol["repair_max_moves"]),
        fallback_max_search_nodes=int(protocol["fallback_max_search_nodes"]),
        enable_repair=bool(post["enable_repair"]),
        enable_fallback=bool(post["enable_fallback"]),
        always_include_fallback=bool(post["always_include_fallback"]),
    )


def _result_payload(result: Any, pool_best: float) -> dict[str, Any]:
    metrics = result.metrics
    raw = metrics.get("best_raw_objective")
    pre = metrics.get("best_pre_fallback_objective")
    return {
        "success": bool(result.success),
        "source": result.source,
        "objective": result.objective,
        "gap_to_pool_best": None if result.objective is None else float(result.objective) / pool_best - 1.0,
        "raw_success": raw is not None,
        "raw_gap_to_pool_best": None if raw is None else float(raw) / pool_best - 1.0,
        "pre_fallback_success": pre is not None,
        "pre_fallback_gap": None if pre is None else float(pre) / pool_best - 1.0,
        "raw_any_feasible": bool(metrics.get("raw_any_feasible", False)),
        "raw_feasible_count": int(metrics.get("raw_feasible_count", 0)),
        "num_raw_proposals": int(metrics.get("num_raw_proposals", 0)),
        "raw_feasible_rate": metrics.get("raw_feasible_rate"),
        "raw_unique_rate": metrics.get("raw_unique_rate"),
        "raw_pairwise_hamming": metrics.get("raw_pairwise_hamming"),
        "raw_capacity_violation_rate": metrics.get("raw_capacity_violation_rate"),
        "raw_link_violation_rate": metrics.get("raw_link_violation_rate"),
        "repair_attempts": int(metrics.get("repair_attempts", 0)),
        "repair_successes": int(metrics.get("repair_successes", 0)),
        "repair_success_rate": float(metrics.get("repair_success_rate", 0.0)),
        "fallback_invoked": bool(metrics.get("fallback_invoked", False)),
        "fallback_success": bool(metrics.get("fallback_success", False)),
        "sampling_seconds": float(metrics.get("sampling_seconds", 0.0)),
        "verification_seconds": float(metrics.get("verification_seconds", 0.0)),
        "repair_seconds": float(metrics.get("repair_seconds", 0.0)),
        "fallback_seconds": float(metrics.get("fallback_seconds", 0.0)),
        "total_seconds": float(metrics.get("total_seconds", 0.0)),
        "masked_model_forwards": metrics.get("masked_model_forwards"),
        "masked_completed_rate": metrics.get("masked_completed_rate"),
    }


def _sample_proposals(
    *,
    method_id: str,
    method: Mapping[str, Any],
    direct_solver: Any,
    masked_solver: Any,
    instance: Any,
    protocol: Mapping[str, Any],
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    family = method["family"]
    if family == "direct":
        proposals, probabilities, seconds = sample_direct_proposals(
            direct_solver.model,
            instance,
            direct_solver.feature_schema,
            config=InferenceConfig(
                num_samples=int(method["samples"]),
                sample_batch_size=min(int(protocol["sample_batch_size"]), int(method["samples"])),
            ),
            device=device,
            generator=generator,
        )
        return {"proposals": proposals, "probabilities": probabilities, "seconds": seconds}
    if family == "masked":
        sampled = sample_masked_proposals(
            masked_solver.model,
            instance,
            masked_solver.schedule,
            masked_solver.feature_schema,
            config=MaskedDecodeConfig(
                num_samples=int(method["samples"]),
                sample_batch_size=min(int(protocol["sample_batch_size"]), int(method["samples"])),
                stochastic=bool(method["stochastic"]),
                temperature=float(protocol["temperature"]),
            ),
            device=device,
            generator=generator,
        )
        return {
            "proposals": sampled.proposals,
            "probabilities": sampled.probabilities,
            "seconds": sampled.sampling_seconds,
            "masked_model_forwards": sampled.model_forwards,
            "masked_completed_rate": float(sampled.completed.mean()),
        }
    if family == "random":
        # sample_random_proposals builds a CPU compatibility tensor, so its
        # generator must also be CPU-backed even when learned methods use CUDA.
        random_generator = torch.Generator(device="cpu").manual_seed(
            int(generator.initial_seed())
        )
        start = perf_counter()
        proposals = sample_random_proposals(
            instance,
            num_samples=int(method["samples"]),
            generator=random_generator,
        )
        return {
            "proposals": proposals,
            "probabilities": None,
            "seconds": perf_counter() - start,
        }
    raise ValueError(f"Unsupported proposal family for {method_id}: {family}")


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


def _parse_seed_list(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return SEEDS
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    unknown = sorted(set(seeds) - set(SEEDS))
    if unknown:
        raise ValueError(f"Unsupported Stage 3.9 seeds: {unknown}")
    return seeds


def run_evaluation(root: Path, *, selected_seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    protocol = _evaluation_protocol()
    protocol_hash = _hash_payload(protocol)
    device = torch.device(protocol["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Stage 3.9 evaluation protocol.")
    seed_everything(int(protocol["evaluation_seed"]), deterministic=bool(protocol["deterministic"]))

    dataset_root = _resolve(root, protocol["dataset_root"])
    freeze = audit_dataset_freeze(dataset_root)
    if int(freeze["dataset_instance_count"]) != int(protocol["expected_instances"]):
        raise ValueError("Sealed dataset instance count disagrees with protocol.")
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=(str(protocol["partition"]),),
        require_freeze=True,
    )
    training_path = _resolve(root, protocol["training_freeze"])
    training = _read_json(training_path)
    if training.get("scope") != TRAINING_FREEZE_SCOPE:
        raise ValueError("Run freeze-training before evaluation.")
    output_root = _resolve(root, protocol["output_root"])
    records_root = output_root / "records"
    dataset_hash = file_sha256(dataset_root / "dataset_freeze.json")
    training_hash = file_sha256(training_path)

    completed = 0
    methods = protocol["methods"]
    modes = protocol["postprocessing_modes"]
    learned_methods = [
        method_id
        for method_id, method in methods.items()
        if method["family"] in {"direct", "masked", "random"}
    ]

    for training_seed in selected_seeds:
        seed_key = str(training_seed)
        direct_entry = training["runs"][seed_key]["direct"]
        masked_entry = training["runs"][seed_key]["masked_conditional"]
        direct = load_stage3_solver(_resolve(root, direct_entry["paths"]["best.pt"]), dataset, device)
        masked = load_stage3_solver(_resolve(root, masked_entry["paths"]["best.pt"]), dataset, device)
        if masked.schedule is None:
            raise ValueError("Masked checkpoint has no absorbing-MASK schedule.")

        for index in range(len(dataset)):
            item = dataset[index]
            record_path = records_root / seed_key / f"{item.instance.instance_id}.json"
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
                sampled = _sample_proposals(
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
                            f"stage39:{method_id}:{training_seed}:{item.instance.instance_id}",
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
                    payload = _result_payload(result, pool_best)
                    if "masked_model_forwards" in sampled:
                        payload["masked_model_forwards"] = int(sampled["masked_model_forwards"])
                        payload["masked_completed_rate"] = float(sampled["masked_completed_rate"])
                    mode_payload[mode] = payload
                method_results[method_id] = mode_payload

            fallback_result = solve_fallback_only(
                item.instance,
                max_search_nodes=int(protocol["fallback_max_search_nodes"]),
            )
            method_results["fallback_only"] = {
                "full": _result_payload(fallback_result, pool_best)
            }
            write_json(
                record_path,
                {
                    "schema_version": "1.0",
                    "scope": RECORD_SCOPE,
                    "training_seed": int(training_seed),
                    "instance_id": item.instance.instance_id,
                    "pool_best": pool_best,
                    "protocol_sha256": protocol_hash,
                    "training_freeze_sha256": training_hash,
                    "dataset_freeze_sha256": dataset_hash,
                    "methods": method_results,
                },
            )
            completed += 1
    return {
        "records_completed": completed,
        "expected": len(dataset) * len(selected_seeds),
        "selected_seeds": list(selected_seeds),
        "output_root": _relative(root, output_root),
    }


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return mean(finite) if finite else None


def _finite_std(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return pstdev(finite) if len(finite) > 1 else 0.0 if finite else None


def _aggregate(records: list[Mapping[str, Any]], method_id: str, mode: str) -> dict[str, Any]:
    rows = [row["methods"][method_id][mode] for row in records if mode in row["methods"][method_id]]
    sources = ("raw", "repair", "fallback", "failure")
    return {
        "records": len(rows),
        "success_rate": mean(float(row["success"]) for row in rows),
        "mean_gap_to_pool_best": _finite_mean([row["gap_to_pool_best"] for row in rows]),
        "gap_std": _finite_std([row["gap_to_pool_best"] for row in rows]),
        "raw_success_rate": mean(float(row["raw_success"]) for row in rows),
        "mean_raw_gap_to_pool_best": _finite_mean([row["raw_gap_to_pool_best"] for row in rows]),
        "pre_fallback_success_rate": mean(float(row["pre_fallback_success"]) for row in rows),
        "mean_pre_fallback_gap": _finite_mean([row["pre_fallback_gap"] for row in rows]),
        "raw_any_feasibility": mean(float(row["raw_any_feasible"]) for row in rows),
        "mean_raw_feasible_rate": _finite_mean([row["raw_feasible_rate"] for row in rows]),
        "mean_raw_unique_rate": _finite_mean([row["raw_unique_rate"] for row in rows]),
        "mean_raw_pairwise_hamming": _finite_mean([row["raw_pairwise_hamming"] for row in rows]),
        "mean_capacity_violation_rate": _finite_mean([row["raw_capacity_violation_rate"] for row in rows]),
        "mean_link_violation_rate": _finite_mean([row["raw_link_violation_rate"] for row in rows]),
        "repair_attempts_mean": mean(float(row["repair_attempts"]) for row in rows),
        "repair_success_rate_mean": mean(float(row["repair_success_rate"]) for row in rows),
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


def _paired(records: list[Mapping[str, Any]], *, direct: str, masked: str, mode: str, stage: str) -> dict[str, Any]:
    by_instance: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_instance[str(row["instance_id"])].append(row)
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


def _relative_improvement(direct_value: float | None, masked_value: float | None) -> float | None:
    if direct_value is None or masked_value is None or direct_value == 0:
        return None
    return (float(direct_value) - float(masked_value)) / float(direct_value)


def _write_report(path: Path, evidence: Mapping[str, Any]) -> None:
    overall = evidence["overall"]
    protocol = evidence["protocol"]
    main = evidence["main_comparison"]
    methods = protocol["methods"]
    lines = [
        "# Phase 6E-E Stage 3.9 Ten-Seed Forward-Budget Evaluation",
        "",
        "This report evaluates the absorbing-MASK solver with ten training seeds.",
        "The primary hard-cost anchor is the same neural forward-equivalent budget",
        "`B_NN = N_prop * N_step`, not wall-clock time matching.",
        "",
        "## Main Anchor",
        "",
        "| Method | B_NN | Raw success | Raw gap | Repair-only gap | Full gap | Source(raw/repair/fallback) | Time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_id in (MAIN_DIRECT, MAIN_MASKED):
        raw = overall[method_id]["raw_only"]
        repair = overall[method_id]["repair_only"]
        full = overall[method_id]["full"]
        source = full["source_rates"]
        lines.append(
            f"| {method_id} | {methods[method_id]['forward_equivalent_budget']} | "
            f"{100 * raw['raw_success_rate']:.2f}% | "
            f"{100 * raw['mean_raw_gap_to_pool_best']:.3f}% | "
            f"{100 * repair['mean_gap_to_pool_best']:.3f}% | "
            f"{100 * full['mean_gap_to_pool_best']:.3f}% | "
            f"{100 * source['raw']:.2f}% / {100 * source['repair']:.2f}% / {100 * source['fallback']:.2f}% | "
            f"{full['mean_total_seconds']:.3f} s |"
        )
    lines.extend([
        "",
        "Main paired tests are aggregated by sealed instance after averaging over",
        "the ten training seeds.",
        "",
        f"- Raw paired wins/losses/ties: {main['raw']['masked_wins']} / {main['raw']['direct_wins']} / {main['raw']['ties']}, p = {main['raw']['sign_test_pvalue']:.6g}.",
        f"- Pre-fallback paired wins/losses/ties: {main['pre']['masked_wins']} / {main['pre']['direct_wins']} / {main['pre']['ties']}, p = {main['pre']['sign_test_pvalue']:.6g}.",
        f"- Final paired wins/losses/ties: {main['final']['masked_wins']} / {main['final']['direct_wins']} / {main['final']['ties']}, p = {main['final']['sign_test_pvalue']:.6g}.",
        "",
        "## Full-Pipeline Baselines",
        "",
        "| Method | B_NN | Success | Final gap | Raw success | Fallback selected | Time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    baseline_order = [
        "fallback_only",
        "random_k64",
        "direct_k64",
        "masked_deterministic_k1",
        "masked_diffusion_k8",
    ]
    for method_id in baseline_order:
        full = overall[method_id]["full"]
        bnn = methods[method_id]["forward_equivalent_budget"]
        lines.append(
            f"| {method_id} | {bnn} | {100 * full['success_rate']:.2f}% | "
            f"{100 * full['mean_gap_to_pool_best']:.3f}% | "
            f"{100 * full['raw_success_rate']:.2f}% | "
            f"{100 * full['source_rates']['fallback']:.2f}% | "
            f"{full['mean_total_seconds']:.3f} s |"
        )
    lines.extend([
        "",
        "## Forward-Budget Sensitivity",
        "",
        "| B_NN | Direct full gap | Masked full gap | Direct raw success | Masked raw success |",
        "|---:|---:|---:|---:|---:|",
    ])
    for direct_k, masked_k in ((8, "masked_deterministic_k1"), (16, "masked_diffusion_k2"), (32, "masked_diffusion_k4"), (64, "masked_diffusion_k8"), (128, "masked_diffusion_k16")):
        direct_id = f"direct_k{direct_k}"
        direct_full = overall[direct_id]["full"]
        masked_full = overall[masked_k]["full"]
        direct_raw = overall[direct_id]["raw_only"]
        masked_raw = overall[masked_k]["raw_only"]
        lines.append(
            f"| {direct_k} | "
            f"{100 * direct_full['mean_gap_to_pool_best']:.3f}% | "
            f"{100 * masked_full['mean_gap_to_pool_best']:.3f}% | "
            f"{100 * direct_raw['raw_success_rate']:.2f}% | "
            f"{100 * masked_raw['raw_success_rate']:.2f}% |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(root: Path) -> dict[str, Any]:
    protocol = _evaluation_protocol()
    protocol_hash = _hash_payload(protocol)
    output_root = _resolve(root, protocol["output_root"])
    dataset_root = _resolve(root, protocol["dataset_root"])
    training_path = _resolve(root, protocol["training_freeze"])
    dataset_hash = file_sha256(dataset_root / "dataset_freeze.json")
    training_hash = file_sha256(training_path)
    records = [
        _read_json(path)
        for path in sorted((output_root / "records").glob("*/*.json"))
    ]
    valid = [
        row for row in records
        if row.get("scope") == RECORD_SCOPE
        and row.get("protocol_sha256") == protocol_hash
        and row.get("training_freeze_sha256") == training_hash
        and row.get("dataset_freeze_sha256") == dataset_hash
    ]
    expected = int(protocol["expected_instances"]) * len(SEEDS)
    if len(valid) != expected:
        raise ValueError(f"Expected {expected} records, found {len(valid)}.")

    methods = protocol["methods"]
    modes = protocol["postprocessing_modes"]
    overall = {
        method_id: {
            mode: _aggregate(valid, method_id, mode)
            for mode in (("full",) if method["family"] == "fallback" else modes.keys())
        }
        for method_id, method in methods.items()
    }
    per_seed = {}
    for seed in SEEDS:
        seed_rows = [row for row in valid if int(row["training_seed"]) == int(seed)]
        per_seed[str(seed)] = {
            method_id: {
                mode: _aggregate(seed_rows, method_id, mode)
                for mode in (("full",) if method["family"] == "fallback" else modes.keys())
            }
            for method_id, method in methods.items()
        }

    direct_full = overall[MAIN_DIRECT]["full"]
    masked_full = overall[MAIN_MASKED]["full"]
    direct_raw = overall[MAIN_DIRECT]["raw_only"]
    masked_raw = overall[MAIN_MASKED]["raw_only"]
    main = {
        "raw": _paired(valid, direct=MAIN_DIRECT, masked=MAIN_MASKED, mode="raw_only", stage="raw"),
        "pre": _paired(valid, direct=MAIN_DIRECT, masked=MAIN_MASKED, mode="full", stage="pre"),
        "final": _paired(valid, direct=MAIN_DIRECT, masked=MAIN_MASKED, mode="full", stage="final"),
        "relative_raw_gap_improvement": _relative_improvement(
            direct_raw["mean_raw_gap_to_pool_best"],
            masked_raw["mean_raw_gap_to_pool_best"],
        ),
        "relative_pre_fallback_gap_improvement": _relative_improvement(
            direct_full["mean_pre_fallback_gap"],
            masked_full["mean_pre_fallback_gap"],
        ),
        "relative_final_gap_improvement": _relative_improvement(
            direct_full["mean_gap_to_pool_best"],
            masked_full["mean_gap_to_pool_best"],
        ),
        "raw_success_rate_improvement_pp": (
            masked_raw["raw_success_rate"] - direct_raw["raw_success_rate"]
        ),
        "final_success_rate_improvement_pp": (
            masked_full["success_rate"] - direct_full["success_rate"]
        ),
    }
    seed_summary = {
        "masked_raw_better": sum(
            per_seed[str(seed)][MAIN_MASKED]["raw_only"]["raw_success_rate"]
            > per_seed[str(seed)][MAIN_DIRECT]["raw_only"]["raw_success_rate"]
            for seed in SEEDS
        ),
        "masked_raw_gap_better": sum(
            per_seed[str(seed)][MAIN_MASKED]["raw_only"]["mean_raw_gap_to_pool_best"]
            < per_seed[str(seed)][MAIN_DIRECT]["raw_only"]["mean_raw_gap_to_pool_best"]
            for seed in SEEDS
        ),
        "masked_pre_gap_better": sum(
            per_seed[str(seed)][MAIN_MASKED]["full"]["mean_pre_fallback_gap"]
            < per_seed[str(seed)][MAIN_DIRECT]["full"]["mean_pre_fallback_gap"]
            for seed in SEEDS
        ),
        "masked_final_gap_better": sum(
            per_seed[str(seed)][MAIN_MASKED]["full"]["mean_gap_to_pool_best"]
            < per_seed[str(seed)][MAIN_DIRECT]["full"]["mean_gap_to_pool_best"]
            for seed in SEEDS
        ),
        "total_seeds": len(SEEDS),
    }
    evidence = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "training_freeze": _relative(root, training_path),
        "training_freeze_sha256": training_hash,
        "dataset_freeze_sha256": dataset_hash,
        "records": len(valid),
        "overall": overall,
        "per_seed": per_seed,
        "main_comparison": main,
        "seed_summary": seed_summary,
        "claim_boundary": (
            "Use same neural forward-equivalent budget as the primary comparison. "
            "Wall-clock runtime is reported as secondary deployment evidence."
        ),
    }
    evidence_path = output_root / "forward_budget_evidence.json"
    report_path = output_root / "forward_budget_report.md"
    write_json(evidence_path, evidence)
    _write_report(report_path, evidence)
    return {
        "evidence": _relative(root, evidence_path),
        "evidence_sha256": file_sha256(evidence_path),
        "report": _relative(root, report_path),
        "records": len(valid),
        "main_raw_p": main["raw"]["sign_test_pvalue"],
        "main_pre_p": main["pre"]["sign_test_pvalue"],
        "main_final_p": main["final"]["sign_test_pvalue"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze-training", "run", "finalize", "all"))
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated subset of Stage 3.9 seeds for sharded run action.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selected_seeds = _parse_seed_list(args.seeds)
    if args.action in {"freeze-training", "all"}:
        print(freeze_training(root))
    if args.action in {"run", "all"}:
        print(run_evaluation(root, selected_seeds=selected_seeds))
    if args.action in {"finalize", "all"}:
        print(finalize(root))


if __name__ == "__main__":
    main()
