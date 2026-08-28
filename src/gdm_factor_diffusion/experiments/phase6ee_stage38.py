"""Sealed multi-seed confirmation for the absorbing-MASK graph solver."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import numpy as np
import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.data.contracts import audit_dataset_config_contract
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    solve_with_direct_predictor,
    solve_with_masked_model,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze

from .phase6ee_stage3 import load_stage3_solver
from .schema import file_sha256


PREPARATION_SCOPE = "phase6e_e_stage38_sealed_preparation"
TRAINING_SCOPE = "phase6e_e_stage38_multiseed_training_freeze"
RECORD_SCOPE = "phase6e_e_stage38_seed_instance_record"
EVIDENCE_SCOPE = "phase6e_e_stage38_sealed_evidence"
DECISION_SCOPE = "phase6e_e_stage38_sealed_decision"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: str | Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _rank(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
    gap = record["mean_pre_fallback_gap"]
    return (
        -float(record["pre_fallback_success_rate"]),
        float("inf") if gap is None else float(gap),
        -float(record["raw_any_feasibility"]),
        float(record["mean_online_seconds"]),
    )


def _training_summary(
    root: Path, run_directory: Path, expected_config: Mapping[str, Any]
) -> dict[str, Any]:
    paths = {
        name: run_directory / name
        for name in ("config.json", "metrics.jsonl", "best.pt", "latest.pt")
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_directory.name}: {missing}")
    if _read_json(paths["config.json"]) != expected_config:
        raise ValueError(f"Config drift in {run_directory.name}.")
    records = _read_jsonl(paths["metrics.jsonl"])
    train = [row for row in records if row.get("type") == "train"]
    selection = [
        row for row in records if row.get("type") == "checkpoint_selection"
    ]
    if not train or not selection:
        raise ValueError(f"Missing records in {run_directory.name}.")
    final_step = int(train[-1]["step"])
    expected_steps = int(expected_config["optimization"]["max_steps"])
    interval = int(expected_config["optimization"]["validation_interval"])
    if final_step != expected_steps or len(selection) != final_step // interval:
        raise ValueError(f"Incomplete selection history in {run_directory.name}.")
    best_record = min(selection, key=_rank)
    best = torch.load(paths["best.pt"], map_location="cpu", weights_only=False)
    latest = torch.load(paths["latest.pt"], map_location="cpu", weights_only=False)
    if int(best["step"]) != int(best_record["step"]):
        raise ValueError(f"Best checkpoint mismatch in {run_directory.name}.")
    if int(latest["step"]) != final_step:
        raise ValueError(f"Latest checkpoint mismatch in {run_directory.name}.")
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


def _training_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment = config["experiment"]
    return {
        "dataset_root": experiment["dataset_root"],
        "train_partition": experiment["train_partition"],
        "checkpoint_partition": experiment["checkpoint_partition"],
        "target_mode": experiment["target_mode"],
        "model": config["model"],
        "direct_control": config["direct_control"],
        "optimization": config["optimization"],
        "checkpoint_selection": config["checkpoint_selection"],
    }


def prepare_stage38(
    config_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    config_path = Path(config_path).resolve()
    confirmation = load_config(config_path)["confirmation"]
    destination = _resolve(root, confirmation["output"]["preparation_lock"])
    if destination.exists():
        return verify_stage38_lock(destination, implementation_root=root)

    authorization_path = _resolve(root, confirmation["authorization_lock"])
    authorization = _read_json(authorization_path)
    if not authorization.get("sealed_multiseed_authorized"):
        raise ValueError("Stage 3.7 did not authorize sealed confirmation.")
    dataset_config_path = _resolve(root, confirmation["sealed_dataset_config"])
    dataset_config = load_config(dataset_config_path)
    audit_dataset_config_contract(dataset_config)
    if int(dataset_config["dataset"]["contract"]["expected_instance_count"]) != int(
        confirmation["expected_instances"]
    ):
        raise ValueError("Sealed dataset count disagrees with the confirmation contract.")
    sealed_root = _resolve(root, confirmation["sealed_dataset_root"])
    if sealed_root.exists():
        raise ValueError("Sealed data already exist before contract preparation.")

    seeds = [int(seed) for seed in confirmation["training_seeds"]]
    config_paths = {
        seed: _resolve(root, confirmation["training_configs"][str(seed)])
        for seed in seeds
    }
    configs = {seed: load_config(path) for seed, path in config_paths.items()}
    reference_protocol = _training_protocol(configs[seeds[0]])
    for seed in seeds:
        if int(configs[seed]["experiment"]["seed"]) != seed:
            raise ValueError(f"Training config seed mismatch for {seed}.")
        if _training_protocol(configs[seed]) != reference_protocol:
            raise ValueError("Training protocols differ across seeds.")

    source_paths = (
        "src/gdm_factor_diffusion/data/contracts.py",
        "src/gdm_factor_diffusion/experiments/phase6ee_stage38.py",
        "src/gdm_factor_diffusion/inference/masked_decode.py",
        "src/gdm_factor_diffusion/inference/solve.py",
        "src/gdm_factor_diffusion/models/conditional_denoiser.py",
        "scripts/64_train_phase6e_e_stage3.py",
        "scripts/72_run_phase6e_e_stage38_sealed.py",
        "active_stage3/run_sealed_multiseed.ps1",
    )
    lock = {
        "schema_version": "1.0",
        "scope": PREPARATION_SCOPE,
        "config": _relative(root, config_path),
        "config_sha256": file_sha256(config_path),
        "authorization_lock": _relative(root, authorization_path),
        "authorization_lock_sha256": file_sha256(authorization_path),
        "development_dataset_root": confirmation["development_dataset_root"],
        "development_dataset_freeze_sha256": file_sha256(
            _resolve(root, confirmation["development_dataset_root"])
            / "dataset_freeze.json"
        ),
        "sealed_dataset_config": _relative(root, dataset_config_path),
        "sealed_dataset_config_sha256": file_sha256(dataset_config_path),
        "sealed_dataset_root": confirmation["sealed_dataset_root"],
        "sealed_partition": confirmation["sealed_partition"],
        "expected_instances": int(confirmation["expected_instances"]),
        "training_seeds": seeds,
        "reused_training_seed": int(confirmation["reused_training_seed"]),
        "training_configs": {
            str(seed): {
                "path": _relative(root, config_paths[seed]),
                "sha256": file_sha256(config_paths[seed]),
            }
            for seed in seeds
        },
        "reused_training_freeze": confirmation["reused_training_freeze"],
        "reused_training_freeze_sha256": file_sha256(
            _resolve(root, confirmation["reused_training_freeze"])
        ),
        "new_training_root": confirmation["new_training_root"],
        "device": confirmation["device"],
        "deterministic": bool(confirmation["deterministic"]),
        "evaluation_seed": int(confirmation["evaluation_seed"]),
        "methods": confirmation["methods"],
        "postprocessing": confirmation["postprocessing"],
        "acceptance": confirmation["acceptance"],
        "output": confirmation["output"],
        "source_sha256": {name: file_sha256(root / name) for name in source_paths},
    }
    write_json(destination, lock)
    return verify_stage38_lock(destination, implementation_root=root)


def verify_stage38_lock(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PREPARATION_SCOPE:
        raise ValueError("Unsupported Stage 3.8 preparation lock.")
    for key, hash_key in (
        ("config", "config_sha256"),
        ("authorization_lock", "authorization_lock_sha256"),
        ("sealed_dataset_config", "sealed_dataset_config_sha256"),
        ("reused_training_freeze", "reused_training_freeze_sha256"),
    ):
        if file_sha256(_resolve(root, lock[key])) != lock[hash_key]:
            raise ValueError(f"Stage 3.8 hash mismatch: {key}.")
    development_freeze = (
        _resolve(root, lock["development_dataset_root"]) / "dataset_freeze.json"
    )
    if file_sha256(development_freeze) != lock["development_dataset_freeze_sha256"]:
        raise ValueError("Stage 3.8 development dataset freeze drifted.")
    for entry in lock["training_configs"].values():
        if file_sha256(_resolve(root, entry["path"])) != entry["sha256"]:
            raise ValueError("Stage 3.8 training config drifted.")
    for relative, expected in lock["source_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Stage 3.8 source drift: {relative}.")
    return lock


def verify_training_open(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = verify_stage38_lock(lock_path, implementation_root=root)
    if _resolve(root, lock["sealed_dataset_root"]).exists():
        raise ValueError("Sealed data were opened before multi-seed training froze.")
    return lock


def finalize_stage38_training(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = verify_training_open(lock_path, implementation_root=root)
    reused_seed = int(lock["reused_training_seed"])
    runs: dict[str, Any] = {}
    for seed in lock["training_seeds"]:
        seed = int(seed)
        config = load_config(_resolve(root, lock["training_configs"][str(seed)]["path"]))
        training_root = (
            _resolve(root, config["output"]["root"])
            if seed != reused_seed
            else root / "artifacts" / "phase6e-e-stage3-training"
        )
        runs[str(seed)] = {
            kind: _training_summary(
                root,
                training_root / config["output"]["run_pattern"].format(model_kind=kind),
                config,
            )
            for kind in ("direct", "masked_conditional")
        }
    freeze = {
        "schema_version": "1.0",
        "scope": TRAINING_SCOPE,
        "preparation_lock": _relative(root, lock_path),
        "preparation_lock_sha256": file_sha256(lock_path),
        "training_seeds": lock["training_seeds"],
        "runs": runs,
        "sealed_data_opened": False,
    }
    destination = _resolve(root, lock["output"]["training_freeze"])
    if destination.exists() and _read_json(destination) != freeze:
        raise ValueError("Existing Stage 3.8 training freeze disagrees with runs.")
    write_json(destination, freeze)
    return {"training_freeze": _relative(root, destination), "sha256": file_sha256(destination)}


def authorize_sealed_data(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = verify_stage38_lock(lock_path, implementation_root=root)
    training_path = _resolve(root, lock["output"]["training_freeze"])
    training = _read_json(training_path)
    if training.get("scope") != TRAINING_SCOPE:
        raise ValueError("Stage 3.8 training is not frozen.")
    if file_sha256(_resolve(root, training["preparation_lock"])) != training["preparation_lock_sha256"]:
        raise ValueError("Stage 3.8 training freeze lost provenance.")
    return {
        "authorized": True,
        "dataset_config": lock["sealed_dataset_config"],
        "dataset_root": lock["sealed_dataset_root"],
        "training_freeze_sha256": file_sha256(training_path),
    }


def _inference_config(lock: Mapping[str, Any], samples: int) -> InferenceConfig:
    post = lock["postprocessing"]
    return InferenceConfig(
        num_samples=samples,
        sample_batch_size=min(int(lock["methods"]["sample_batch_size"]), samples),
        repair_max_moves=int(post["repair_max_moves"]),
        fallback_max_search_nodes=int(post["fallback_max_search_nodes"]),
        enable_repair=True,
        enable_fallback=True,
        always_include_fallback=bool(post["always_include_fallback"]),
    )


def _payload(result: Any, pool_best: float) -> dict[str, Any]:
    metrics = result.metrics
    pre = metrics.get("best_pre_fallback_objective")
    return {
        "success": bool(result.success),
        "source": result.source,
        "objective": result.objective,
        "gap_to_pool_best": None if result.objective is None else float(result.objective) / pool_best - 1.0,
        "pre_fallback_success": pre is not None,
        "pre_fallback_gap": None if pre is None else float(pre) / pool_best - 1.0,
        "raw_any_feasible": bool(metrics["raw_any_feasible"]),
        "raw_feasible_rate": float(metrics["raw_feasible_rate"]),
        "fallback_invoked": bool(metrics["fallback_invoked"]),
        "sampling_seconds": float(metrics["sampling_seconds"]),
        "total_seconds": float(metrics["total_seconds"]),
    }


def _load_confirmed_inputs(root: Path, lock: Mapping[str, Any]):
    authorization = authorize_sealed_data(
        _resolve(root, lock["output"]["preparation_lock"]), implementation_root=root
    )
    dataset_root = _resolve(root, authorization["dataset_root"])
    freeze = audit_dataset_freeze(dataset_root)
    if int(freeze["dataset_instance_count"]) != int(lock["expected_instances"]):
        raise ValueError("Stage 3.8 sealed dataset has the wrong size.")
    manifest = _read_json(dataset_root / "manifest.json")
    if set(entry["partition"] for entry in manifest["instances"]) != {lock["sealed_partition"]}:
        raise ValueError("Stage 3.8 sealed partition drifted.")
    config = load_config(_resolve(root, lock["sealed_dataset_config"]))["dataset"].copy()
    config.pop("output", None)
    if manifest["generation_config"] != config:
        raise ValueError("Stage 3.8 generated data disagree with the locked config.")
    training_path = _resolve(root, lock["output"]["training_freeze"])
    training = _read_json(training_path)
    return dataset_root, file_sha256(dataset_root / "dataset_freeze.json"), training_path, training


def run_stage38_evaluation(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_stage38_lock(lock_path, implementation_root=root)
    device = torch.device(lock["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Stage 3.8 contract.")
    seed_everything(int(lock["evaluation_seed"]), deterministic=bool(lock["deterministic"]))
    dataset_root, dataset_hash, training_path, training = _load_confirmed_inputs(root, lock)
    dataset = LabeledDeploymentDataset(
        dataset_root, partitions=(lock["sealed_partition"],), require_freeze=True
    )
    output_root = _resolve(root, lock["output"]["evaluation_root"])
    completed = 0
    methods = lock["methods"]
    for training_seed in lock["training_seeds"]:
        seed_key = str(training_seed)
        direct_entry = training["runs"][seed_key]["direct"]
        masked_entry = training["runs"][seed_key]["masked_conditional"]
        direct_path = _resolve(root, direct_entry["paths"]["best.pt"])
        masked_path = _resolve(root, masked_entry["paths"]["best.pt"])
        direct = load_stage3_solver(direct_path, dataset, device)
        masked = load_stage3_solver(masked_path, dataset, device)
        if masked.schedule is None:
            raise ValueError("Stage 3.8 masked checkpoint has no schedule.")

        warm = dataset[0]
        solve_with_direct_predictor(
            direct.model, warm.instance, direct.feature_schema,
            config=_inference_config(lock, int(methods["direct_samples"])), device=device,
            generator=torch.Generator(device=device).manual_seed(
                derive_seed(int(lock["evaluation_seed"]), f"warm:direct:{training_seed}")
            ),
        )
        for samples, stochastic in ((1, False), (int(methods["diffusion_samples"]), True)):
            solve_with_masked_model(
                masked.model, warm.instance, masked.schedule, masked.feature_schema,
                decode_config=MaskedDecodeConfig(
                    num_samples=samples,
                    sample_batch_size=min(int(methods["sample_batch_size"]), samples),
                    stochastic=stochastic,
                    temperature=float(methods["temperature"]),
                ),
                inference_config=_inference_config(lock, samples), device=device,
                generator=torch.Generator(device=device).manual_seed(
                    derive_seed(int(lock["evaluation_seed"]), f"warm:masked:{training_seed}:{samples}")
                ),
            )

        for index in range(len(dataset)):
            item = dataset[index]
            record_path = output_root / "records" / seed_key / f"{item.instance.instance_id}.json"
            if record_path.exists():
                row = _read_json(record_path)
                if (
                    row.get("preparation_lock_sha256") == file_sha256(lock_path)
                    and row.get("dataset_freeze_sha256") == dataset_hash
                    and row.get("training_freeze_sha256") == file_sha256(training_path)
                ):
                    completed += 1
                    continue
                raise ValueError(f"Stale Stage 3.8 record: {record_path}")
            pool_best = float(np.min(item.pool.latencies))
            direct_result = solve_with_direct_predictor(
                direct.model, item.instance, direct.feature_schema,
                config=_inference_config(lock, int(methods["direct_samples"])), device=device,
                generator=torch.Generator(device=device).manual_seed(
                    derive_seed(int(lock["evaluation_seed"]), f"direct:{training_seed}:{item.instance.instance_id}")
                ),
            )
            results = {"direct_k64": _payload(direct_result, pool_best)}
            for method_id, samples, stochastic in (
                ("masked_deterministic_k1", 1, False),
                ("masked_diffusion_k8", int(methods["diffusion_samples"]), True),
            ):
                result = solve_with_masked_model(
                    masked.model, item.instance, masked.schedule, masked.feature_schema,
                    decode_config=MaskedDecodeConfig(
                        num_samples=samples,
                        sample_batch_size=min(int(methods["sample_batch_size"]), samples),
                        stochastic=stochastic,
                        temperature=float(methods["temperature"]),
                    ),
                    inference_config=_inference_config(lock, samples), device=device,
                    generator=torch.Generator(device=device).manual_seed(
                        derive_seed(int(lock["evaluation_seed"]), f"{method_id}:{training_seed}:{item.instance.instance_id}")
                    ),
                )
                results[method_id] = _payload(result, pool_best)
            write_json(record_path, {
                "schema_version": "1.0",
                "scope": RECORD_SCOPE,
                "training_seed": int(training_seed),
                "instance_id": item.instance.instance_id,
                "pool_best": pool_best,
                "preparation_lock_sha256": file_sha256(lock_path),
                "dataset_freeze_sha256": dataset_hash,
                "training_freeze_sha256": file_sha256(training_path),
                "methods": results,
            })
            completed += 1
    return {"records_completed": completed, "expected": len(dataset) * len(lock["training_seeds"])}


def _aggregate(records: list[Mapping[str, Any]], method_id: str) -> dict[str, Any]:
    rows = [record["methods"][method_id] for record in records]
    pre = [float(row["pre_fallback_gap"]) for row in rows if row["pre_fallback_success"]]
    final = [float(row["gap_to_pool_best"]) for row in rows if row["success"]]
    return {
        "records": len(rows),
        "pre_fallback_success_rate": mean(float(row["pre_fallback_success"]) for row in rows),
        "mean_pre_fallback_gap": mean(pre) if pre else None,
        "raw_any_feasibility": mean(float(row["raw_any_feasible"]) for row in rows),
        "final_success_rate": mean(float(row["success"]) for row in rows),
        "mean_gap_to_pool_best": mean(final) if final else None,
        "fallback_invocation_rate": mean(float(row["fallback_invoked"]) for row in rows),
        "mean_sampling_seconds": mean(float(row["sampling_seconds"]) for row in rows),
        "mean_total_seconds": mean(float(row["total_seconds"]) for row in rows),
    }


def _sign_test_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def evaluate_stage38_gate(
    records: list[Mapping[str, Any]],
    training_seeds: list[int],
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    methods = ("direct_k64", "masked_deterministic_k1", "masked_diffusion_k8")
    per_seed: dict[str, Any] = {}
    for seed in training_seeds:
        seed_rows = [row for row in records if int(row["training_seed"]) == int(seed)]
        per_seed[str(seed)] = {method: _aggregate(seed_rows, method) for method in methods}
    overall = {method: _aggregate(records, method) for method in methods}
    direct = overall["direct_k64"]
    diffusion = overall["masked_diffusion_k8"]
    relative = (
        (direct["mean_pre_fallback_gap"] - diffusion["mean_pre_fallback_gap"])
        / direct["mean_pre_fallback_gap"]
    )

    by_instance: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_instance[str(row["instance_id"])].append(row)
    wins = losses = ties = 0
    for rows in by_instance.values():
        scores = {}
        for method in ("direct_k64", "masked_diffusion_k8"):
            values = [row["methods"][method] for row in rows]
            success = mean(float(value["pre_fallback_success"]) for value in values)
            gaps = [float(value["pre_fallback_gap"]) for value in values if value["pre_fallback_success"]]
            scores[method] = (-success, mean(gaps) if gaps else float("inf"))
        if scores["masked_diffusion_k8"] < scores["direct_k64"]:
            wins += 1
        elif scores["direct_k64"] < scores["masked_diffusion_k8"]:
            losses += 1
        else:
            ties += 1
    pvalue = _sign_test_pvalue(wins, losses)
    time_ratios = {
        seed: values["masked_diffusion_k8"]["mean_total_seconds"]
        / values["direct_k64"]["mean_total_seconds"]
        for seed, values in per_seed.items()
    }
    seed_improvements = {
        seed: (
            values["direct_k64"]["mean_pre_fallback_gap"]
            - values["masked_diffusion_k8"]["mean_pre_fallback_gap"]
        ) / values["direct_k64"]["mean_pre_fallback_gap"]
        for seed, values in per_seed.items()
    }
    checks = {
        "time_comparable_each_seed": all(
            float(acceptance["minimum_time_ratio"]) <= ratio <= float(acceptance["maximum_time_ratio"])
            for ratio in time_ratios.values()
        ),
        "aggregate_gap_improved": relative >= float(acceptance["minimum_aggregate_relative_pre_fallback_gap_improvement"]),
        "positive_gap_improvement_each_seed": all(value > 0 for value in seed_improvements.values()),
        "pre_fallback_success_not_reduced": diffusion["pre_fallback_success_rate"] >= direct["pre_fallback_success_rate"],
        "raw_feasibility_not_reduced": diffusion["raw_any_feasibility"] >= direct["raw_any_feasibility"],
        "final_gap_not_worse": diffusion["mean_gap_to_pool_best"] <= direct["mean_gap_to_pool_best"],
        "final_success_not_reduced": diffusion["final_success_rate"] >= direct["final_success_rate"],
        "instance_wins_exceed_losses": wins > losses,
        "instance_sign_test_passes": pvalue <= float(acceptance["maximum_instance_sign_test_pvalue"]),
    }
    enabled = {
        "positive_gap_improvement_each_seed": bool(acceptance["require_positive_pre_fallback_improvement_each_seed"]),
        "pre_fallback_success_not_reduced": bool(acceptance["require_pre_fallback_success_not_reduced"]),
        "raw_feasibility_not_reduced": bool(acceptance["require_raw_feasibility_not_reduced"]),
        "final_gap_not_worse": bool(acceptance["require_final_gap_not_worse"]),
        "final_success_not_reduced": bool(acceptance["require_final_success_not_reduced"]),
        "instance_wins_exceed_losses": bool(acceptance["require_more_instance_wins_than_losses"]),
    }
    required = ["time_comparable_each_seed", "aggregate_gap_improved", "instance_sign_test_passes"]
    required.extend(name for name, active in enabled.items() if active)
    seed_metric_summary = {
        method: {
            metric: {
                "mean": mean(per_seed[str(seed)][method][metric] for seed in training_seeds),
                "std": pstdev(per_seed[str(seed)][method][metric] for seed in training_seeds),
            }
            for metric in ("mean_pre_fallback_gap", "raw_any_feasibility", "mean_gap_to_pool_best", "mean_total_seconds")
        }
        for method in methods
    }
    return {
        "per_seed": per_seed,
        "overall": overall,
        "seed_metric_summary": seed_metric_summary,
        "time_ratios": time_ratios,
        "relative_pre_fallback_gap_improvements": seed_improvements,
        "aggregate_relative_pre_fallback_gap_improvement": relative,
        "instance_paired_wins": wins,
        "instance_paired_losses": losses,
        "instance_paired_ties": ties,
        "instance_sign_test_pvalue": pvalue,
        "checks": checks,
        "passed": all(checks[name] for name in required),
    }


def finalize_stage38(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_stage38_lock(lock_path, implementation_root=root)
    dataset_root, dataset_hash, training_path, _ = _load_confirmed_inputs(root, lock)
    manifest = _read_json(dataset_root / "manifest.json")
    ids = [str(entry["instance_id"]) for entry in manifest["instances"]]
    output_root = _resolve(root, lock["output"]["evaluation_root"])
    records = []
    record_hashes = {}
    for seed in lock["training_seeds"]:
        for instance_id in ids:
            path = output_root / "records" / str(seed) / f"{instance_id}.json"
            row = _read_json(path)
            if row.get("scope") != RECORD_SCOPE:
                raise ValueError("Invalid Stage 3.8 record scope.")
            records.append(row)
            record_hashes[f"{seed}/{instance_id}"] = file_sha256(path)
    gate = evaluate_stage38_gate(records, [int(seed) for seed in lock["training_seeds"]], lock["acceptance"])
    evidence = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "preparation_lock": _relative(root, lock_path),
        "preparation_lock_sha256": file_sha256(lock_path),
        "training_freeze": _relative(root, training_path),
        "training_freeze_sha256": file_sha256(training_path),
        "dataset_freeze_sha256": dataset_hash,
        "records": len(records),
        "record_sha256": record_hashes,
        "gate": gate,
    }
    evidence_path = _resolve(root, lock["output"]["final_evidence"])
    write_json(evidence_path, evidence)
    decision = {
        "schema_version": "1.0",
        "scope": DECISION_SCOPE,
        "final_evidence": _relative(root, evidence_path),
        "final_evidence_sha256": file_sha256(evidence_path),
        "diffusion_claim_confirmed": gate["passed"],
        "paper_claim_if_confirmed": (
            "graph diffusion improves deployment quality over direct prediction "
            "under a comparable total online budget"
        ),
    }
    decision_path = _resolve(root, lock["output"]["decision_lock"])
    write_json(decision_path, decision)
    return {
        "evidence": _relative(root, evidence_path),
        "evidence_sha256": file_sha256(evidence_path),
        "decision_lock": _relative(root, decision_path),
        "decision_lock_sha256": file_sha256(decision_path),
        "diffusion_claim_confirmed": gate["passed"],
    }
