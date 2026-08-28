"""Locked calibration and one-time pilot contracts for Phase 6E-E Stage 3."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.data.contracts import audit_dataset_config_contract
from gdm_factor_diffusion.diffusion import AbsorbingMaskSchedule
from gdm_factor_diffusion.graph import GraphFeatureSchema, build_factor_graph_batch
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    solve_with_direct_predictor,
    solve_with_masked_model,
)
from gdm_factor_diffusion.models import (
    ConditionalDenoiserConfig,
    DirectPredictorConfig,
    TypedFactorConditionalDenoiser,
    TypedFactorDirectPredictor,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze

from .phase6ee_stage2a import run_trajectory_diffusion_methods
from .runtime import load_learned_solver
from .schema import file_sha256
from .training_aggregation import verify_checkpoint_freeze

PREPARATION_SCOPE = "phase6e_e_stage3_pilot_preparation"
CALIBRATION_SCOPE = "phase6e_e_stage3_checkpoint_calibration"
PILOT_CONTRACT_SCOPE = "phase6e_e_stage3_one_time_pilot_contract"


def prepare_stage3_contract(implementation_root: str | Path) -> dict[str, Any]:
    """Create or read the pre-training Stage 3 development contract.

    This compatibility entry point predates the pilot-calibration routines in
    this module.  Once written, its lock is immutable: later source changes
    must not silently rewrite the evidence that formal training consumed.
    """
    root = Path(implementation_root).resolve()
    contract_path = root / "configs" / "phase6e_e_stage3_contract.yaml"
    destination = (
        root / "artifacts" / "phase6e-e-stage3" / "stage3_contract_lock.json"
    )
    if destination.exists():
        lock = _read_json(destination)
        final_root = _resolve(root, lock["protocol"]["future_final_contract"]["output"])
        if lock.get("final_data_exists") or final_root.exists():
            raise ValueError("Stage 3 final data was unexpectedly opened.")
        return lock

    protocol = load_config(contract_path)["stage3_contract"]
    dataset_config_path = _resolve(root, protocol["development_dataset_config"])
    training_config_path = _resolve(root, protocol["training_config"])
    dataset_config = load_config(dataset_config_path)
    training_config = load_config(training_config_path)
    audit_dataset_config_contract(dataset_config)

    final_root = _resolve(root, protocol["future_final_contract"]["output"])
    if final_root.exists():
        raise ValueError("Stage 3 final data already exists; contract freeze refused.")

    prior_paths = {
        "final_evidence_freeze.json": (
            root
            / "artifacts"
            / "phase6e-e-stage2b"
            / "final_evidence_freeze.json"
        ),
        "stage2b_lock.json": (
            root / "artifacts" / "phase6e-e-stage2b" / "stage2b_lock.json"
        ),
    }
    tracked_paths = {
        "PHASE6E_E_STAGE3_PLAN.md": root / "PHASE6E_E_STAGE3_PLAN.md",
        "configs/dataset_phase6e_e_stage3_development.yaml": dataset_config_path,
        "configs/phase6e_e_stage3_contract.yaml": contract_path,
        "configs/training_phase6e_e_stage3_pilot.yaml": training_config_path,
    }
    lock = {
        "schema_version": "1.0",
        "phase": "6E-E Stage 3 pre-training",
        "protocol": protocol,
        "development_instance_count": int(
            dataset_config["dataset"]["contract"]["expected_instance_count"]
        ),
        "training_seed": int(training_config["experiment"]["seed"]),
        "final_data_exists": False,
        "prior_frozen_sha256": {
            name: file_sha256(path) for name, path in prior_paths.items()
        },
        "file_sha256": {
            name: file_sha256(path) for name, path in tracked_paths.items()
        },
    }
    write_json(destination, lock)
    return lock


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _feature_schema(payload: Mapping[str, Any]) -> GraphFeatureSchema:
    return GraphFeatureSchema(
        service_feature_names=tuple(payload["service_feature_names"]),
        device_feature_names=tuple(payload["device_feature_names"]),
        resource_names=tuple(payload["resource_names"]),
    )


@dataclass(frozen=True, slots=True)
class LoadedStage3Solver:
    model: nn.Module
    feature_schema: GraphFeatureSchema
    schedule: AbsorbingMaskSchedule | None
    model_kind: str


def load_stage3_solver(
    checkpoint_path: str | Path,
    dataset: LabeledDeploymentDataset,
    device: torch.device | str,
) -> LoadedStage3Solver:
    target = torch.device(device)
    payload = torch.load(Path(checkpoint_path), map_location=target, weights_only=True)
    metadata = payload["metadata"]
    schema = _feature_schema(metadata["feature_schema"])
    reference = build_factor_graph_batch(
        [dataset[0].instance], feature_schema=schema
    ).to(target)
    config = metadata["config"]
    kind = str(payload["model_kind"])
    schedule: AbsorbingMaskSchedule | None = None
    if kind == "direct":
        model_config = config["direct_control"]
        model = TypedFactorDirectPredictor.from_batch(
            reference,
            DirectPredictorConfig(
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["num_layers"]),
                dropout=float(model_config["dropout"]),
            ),
        ).to(target)
    elif kind == "masked_conditional":
        model_config = config["model"]
        schedule = AbsorbingMaskSchedule(**payload["mask_schedule"])
        model = TypedFactorConditionalDenoiser.from_batch(
            reference,
            ConditionalDenoiserConfig(
                num_mask_steps=schedule.num_steps,
                hidden_dim=int(model_config["hidden_dim"]),
                num_layers=int(model_config["num_layers"]),
                dropout=float(model_config["dropout"]),
            ),
        ).to(target)
    else:
        raise ValueError(f"Unsupported Stage 3 checkpoint kind: {kind!r}.")
    model.load_state_dict(payload["model"])
    model.eval()
    return LoadedStage3Solver(model, schema, schedule, kind)


def _canonical_historical_checkpoint(
    freeze: Mapping[str, Any], root: Path, seed: int
) -> Path:
    verified = verify_checkpoint_freeze(
        _resolve(root, freeze["_path"])
    ) if "_path" in freeze else freeze
    matches = [entry for entry in verified["runs"] if int(entry["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected one historical checkpoint for seed {seed}.")
    return _resolve(root, matches[0]["best_checkpoint"])


def prepare_stage3_pilot(
    config_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    evaluation = config["evaluation"]
    output = config["output"]
    pilot_root = _resolve(root, output["pilot_root"])
    if pilot_root.exists():
        raise ValueError("Stage 3 pilot output already exists; preparation refused.")
    dataset_root = _resolve(root, evaluation["dataset_root"])
    dataset_freeze = dataset_root / "dataset_freeze.json"
    audit_dataset_freeze(dataset_root)
    training_freeze_path = _resolve(root, evaluation["training_freeze"])
    training_freeze = _read_json(training_freeze_path)
    if not training_freeze.get("formal_training_complete"):
        raise ValueError("Stage 3 formal training is not frozen as complete.")
    if training_freeze.get("pilot_opened"):
        raise ValueError("Training freeze reports that the pilot was opened.")

    direct_path = _resolve(root, evaluation["direct_checkpoint"])
    masked_path = _resolve(root, evaluation["masked_checkpoint"])
    expected = {
        "direct": training_freeze["runs"]["direct"]["sha256"]["best.pt"],
        "masked_conditional": training_freeze["runs"]["masked_conditional"][
            "sha256"
        ]["best.pt"],
    }
    if file_sha256(direct_path).upper() != expected["direct"].upper():
        raise ValueError("Direct checkpoint differs from the training freeze.")
    if file_sha256(masked_path).upper() != expected["masked_conditional"].upper():
        raise ValueError("Masked checkpoint differs from the training freeze.")

    historical_freeze_path = _resolve(
        root, config["historical_anchors"]["checkpoint_freeze"]
    )
    historical_freeze = verify_checkpoint_freeze(historical_freeze_path)
    historical_seed = int(config["historical_anchors"]["seed"])
    historical_path = _canonical_historical_checkpoint(
        historical_freeze, root, historical_seed
    )

    manifest = _read_json(dataset_root / "manifest.json")
    by_partition: dict[str, list[str]] = defaultdict(list)
    for entry in manifest["instances"]:
        by_partition[str(entry["partition"])].append(str(entry["instance_id"]))
    checkpoint_ids = by_partition[str(evaluation["checkpoint_partition"])]
    pilot_ids = by_partition[str(evaluation["pilot_partition"])]
    if len(checkpoint_ids) != 64 or len(pilot_ids) != 64 or set(checkpoint_ids) & set(pilot_ids):
        raise ValueError("Stage 3 checkpoint/pilot partitions are not locked 64/64 disjoint sets.")

    source_paths = (
        "src/gdm_factor_diffusion/experiments/phase6ee_stage3.py",
        "src/gdm_factor_diffusion/inference/masked_decode.py",
        "src/gdm_factor_diffusion/inference/solve.py",
        "src/gdm_factor_diffusion/models/conditional_denoiser.py",
        "scripts/67_prepare_phase6e_e_stage3_pilot.py",
        "scripts/68_calibrate_phase6e_e_stage3_pilot.py",
    )

    lock = {
        "schema_version": "1.0",
        "scope": PREPARATION_SCOPE,
        "config": _relative(root, config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_root": _relative(root, dataset_root),
        "dataset_freeze": _relative(root, dataset_freeze),
        "dataset_freeze_sha256": file_sha256(dataset_freeze),
        "training_freeze": _relative(root, training_freeze_path),
        "training_freeze_sha256": file_sha256(training_freeze_path),
        "checkpoints": {
            "direct": {
                "path": _relative(root, direct_path),
                "sha256": file_sha256(direct_path),
            },
            "masked_conditional": {
                "path": _relative(root, masked_path),
                "sha256": file_sha256(masked_path),
            },
            "historical_diffusion": {
                "seed": historical_seed,
                "path": _relative(root, historical_path),
                "sha256": file_sha256(historical_path),
                "freeze": _relative(root, historical_freeze_path),
                "freeze_sha256": file_sha256(historical_freeze_path),
            },
        },
        "checkpoint_partition": str(evaluation["checkpoint_partition"]),
        "checkpoint_instance_ids": checkpoint_ids,
        "pilot_partition": str(evaluation["pilot_partition"]),
        "pilot_instance_ids": pilot_ids,
        "calibration": config["calibration"],
        "postprocessing": config["postprocessing"],
        "historical_anchors": config["historical_anchors"],
        "gates": config["gates"],
        "device": str(evaluation["device"]),
        "deterministic": bool(evaluation["deterministic"]),
        "seed": int(evaluation["seed"]),
        "calibration_root": str(output["calibration_root"]),
        "pilot_root": str(output["pilot_root"]),
        "pilot_contract_lock": str(output["pilot_contract_lock"]),
        "source_sha256": {
            name: file_sha256(root / name) for name in source_paths
        },
    }
    destination = _resolve(root, output["preparation_lock"])
    write_json(destination, lock)
    return verify_stage3_preparation_lock(destination, implementation_root=root)


def verify_stage3_preparation_lock(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PREPARATION_SCOPE:
        raise ValueError("Unsupported Stage 3 preparation lock.")
    for path_key, hash_key in (
        ("config", "config_sha256"),
        ("dataset_freeze", "dataset_freeze_sha256"),
        ("training_freeze", "training_freeze_sha256"),
    ):
        if file_sha256(_resolve(root, lock[path_key])) != lock[hash_key]:
            raise ValueError(f"Stage 3 lock hash mismatch: {path_key}.")
    for entry in lock["checkpoints"].values():
        if file_sha256(_resolve(root, entry["path"])) != entry["sha256"]:
            raise ValueError("Stage 3 checkpoint hash mismatch.")
    for relative, expected in lock["source_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Stage 3 source hash mismatch: {relative}.")
    if set(lock["checkpoint_instance_ids"]) & set(lock["pilot_instance_ids"]):
        raise ValueError("Stage 3 calibration and pilot IDs overlap.")
    if _resolve(root, lock["pilot_root"]).exists():
        raise ValueError("Stage 3 pilot output already exists.")
    return lock


def _inference_config(lock: Mapping[str, Any], num_samples: int) -> InferenceConfig:
    post = lock["postprocessing"]
    return InferenceConfig(
        num_samples=num_samples,
        sample_batch_size=min(
            int(lock["calibration"]["direct_sample_batch_size"]), num_samples
        ),
        repair_max_moves=int(post["repair_max_moves"]),
        fallback_max_search_nodes=int(post["fallback_max_search_nodes"]),
        enable_repair=True,
        enable_fallback=True,
        always_include_fallback=bool(post["always_include_fallback"]),
    )


def _payload(result, pool_best: float) -> dict[str, Any]:
    metrics = result.metrics
    pre = metrics.get("best_pre_fallback_objective")
    return {
        "success": result.success,
        "source": result.source,
        "objective": result.objective,
        "gap_to_pool_best": None if result.objective is None else result.objective / pool_best - 1.0,
        "pre_fallback_success": pre is not None,
        "pre_fallback_gap": None if pre is None else float(pre) / pool_best - 1.0,
        "raw_any_feasible": bool(metrics["raw_any_feasible"]),
        "raw_feasible_rate": metrics["raw_feasible_rate"],
        "fallback_invoked": bool(metrics["fallback_invoked"]),
        "sampling_seconds": float(metrics["sampling_seconds"]),
        "total_seconds": float(metrics["total_seconds"]),
        "selected_source": result.source,
    }


def _historical_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    anchor = lock["historical_anchors"]
    return {
        "variants": [
            {
                "method_id": "diffusion_final_k4",
                "anchor_indices": [],
                "repair_candidate_limit": None,
            },
            {
                "method_id": "rescue_all_five_b12",
                "anchor_indices": list(anchor["rescue_anchor_indices"]),
                "repair_candidate_limit": int(anchor["rescue_repair_candidate_limit"]),
            },
        ],
        "diffusion": {
            "num_samples": int(anchor["diffusion_samples"]),
            "sample_batch_size": int(anchor["diffusion_sample_batch_size"]),
            "reverse_steps": int(anchor["diffusion_reverse_steps"]),
        },
        "direct": {"sample_batch_size": 1},
        "postprocessing": dict(lock["postprocessing"]),
    }


def run_stage3_calibration(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
    limit: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_stage3_preparation_lock(lock_path, implementation_root=root)
    if smoke and (limit is None or limit > 2):
        raise ValueError("Smoke calibration is limited to at most two instances.")
    if not smoke and limit is not None:
        raise ValueError("The formal calibration must use all locked instances.")
    device = torch.device(lock["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Stage 3 pilot contract.")
    seed_everything(int(lock["seed"]), deterministic=bool(lock["deterministic"]))
    dataset = LabeledDeploymentDataset(
        _resolve(root, lock["dataset_root"]),
        partitions=(lock["checkpoint_partition"],),
        require_freeze=True,
    )
    count = len(dataset) if limit is None else min(limit, len(dataset))
    direct_solver = load_stage3_solver(
        _resolve(root, lock["checkpoints"]["direct"]["path"]), dataset, device
    )
    masked_solver = load_stage3_solver(
        _resolve(root, lock["checkpoints"]["masked_conditional"]["path"]),
        dataset,
        device,
    )
    historical_solver = load_learned_solver(
        _resolve(root, lock["checkpoints"]["historical_diffusion"]["path"]),
        dataset,
        device,
    )
    if direct_solver.model_kind != "direct" or masked_solver.schedule is None:
        raise ValueError("Stage 3 checkpoint families are inconsistent.")

    # Warm CUDA kernels outside the recorded timing scope.
    warm_item = dataset[0]
    warm_pool_best = float(np.min(warm_item.pool.latencies))
    warm_direct = solve_with_direct_predictor(
        direct_solver.model,
        warm_item.instance,
        direct_solver.feature_schema,
        config=_inference_config(lock, 1),
        device=device,
        generator=torch.Generator(device=device).manual_seed(
            derive_seed(int(lock["seed"]), "calibration-warm-direct")
        ),
    )
    _payload(warm_direct, warm_pool_best)
    for warm_stochastic, warm_samples in ((False, 1), (True, 8)):
        warm_inference = _inference_config(lock, warm_samples)
        warm_masked = solve_with_masked_model(
            masked_solver.model,
            warm_item.instance,
            masked_solver.schedule,
            masked_solver.feature_schema,
            decode_config=MaskedDecodeConfig(
                num_samples=warm_samples,
                sample_batch_size=warm_samples,
                stochastic=warm_stochastic,
                temperature=float(lock["calibration"]["temperature"]),
            ),
            inference_config=warm_inference,
            device=device,
            generator=torch.Generator(device=device).manual_seed(
                derive_seed(
                    int(lock["seed"]),
                    f"calibration-warm-masked-{warm_stochastic}",
                )
            ),
        )
        _payload(warm_masked, warm_pool_best)
    run_trajectory_diffusion_methods(
        _historical_lock(lock),
        historical_solver,
        warm_item,
        selected_variant="rescue_all_five_b12",
        generator=torch.Generator(device=device).manual_seed(
            derive_seed(int(lock["seed"]), "calibration-warm-historical")
        ),
        device=device,
    )

    output_root = _resolve(root, lock["calibration_root"])
    if smoke:
        output_root = output_root.parent / f"{output_root.name}-smoke"
    record_root = output_root / "records"
    methods_expected = {
        *(f"direct_k{int(k)}" for k in lock["calibration"]["direct_proposal_grid"]),
        "masked_deterministic_k1",
        f"masked_stochastic_k{int(lock['calibration']['stochastic_masked_samples'])}",
        "diffusion_final_k4",
        "rescue_all_five_b12",
    }
    completed = 0
    for index in range(count):
        item = dataset[index]
        path = record_root / f"{item.instance.instance_id}.json"
        if path.exists():
            existing = _read_json(path)
            if (
                existing.get("lock_sha256") == file_sha256(lock_path)
                and set(existing.get("methods", {})) == methods_expected
            ):
                completed += 1
                continue
            raise ValueError(f"Stale Stage 3 calibration record: {path}")
        pool_best = float(np.min(item.pool.latencies))
        methods: dict[str, Any] = {}
        for proposal_count in lock["calibration"]["direct_proposal_grid"]:
            proposal_count = int(proposal_count)
            method_id = f"direct_k{proposal_count}"
            generator = torch.Generator(device=device).manual_seed(
                derive_seed(int(lock["seed"]), f"{method_id}:{item.instance.instance_id}")
            )
            result = solve_with_direct_predictor(
                direct_solver.model,
                item.instance,
                direct_solver.feature_schema,
                config=_inference_config(lock, proposal_count),
                device=device,
                generator=generator,
            )
            methods[method_id] = _payload(result, pool_best)

        masked_specs = (
            ("masked_deterministic_k1", 1, False),
            (
                f"masked_stochastic_k{int(lock['calibration']['stochastic_masked_samples'])}",
                int(lock["calibration"]["stochastic_masked_samples"]),
                True,
            ),
        )
        for method_id, proposal_count, stochastic in masked_specs:
            generator = torch.Generator(device=device).manual_seed(
                derive_seed(int(lock["seed"]), f"{method_id}:{item.instance.instance_id}")
            )
            inference = _inference_config(lock, proposal_count)
            result = solve_with_masked_model(
                masked_solver.model,
                item.instance,
                masked_solver.schedule,
                masked_solver.feature_schema,
                decode_config=MaskedDecodeConfig(
                    num_samples=proposal_count,
                    sample_batch_size=min(
                        int(lock["calibration"]["masked_sample_batch_size"]),
                        proposal_count,
                    ),
                    stochastic=stochastic,
                    temperature=float(lock["calibration"]["temperature"]),
                ),
                inference_config=inference,
                device=device,
                generator=generator,
            )
            methods[method_id] = _payload(result, pool_best)

        historical_seed = derive_seed(
            int(lock["seed"]), f"historical:{item.instance.instance_id}"
        )
        historical = run_trajectory_diffusion_methods(
            _historical_lock(lock),
            historical_solver,
            item,
            selected_variant="rescue_all_five_b12",
            generator=torch.Generator(device=device).manual_seed(historical_seed),
            device=device,
        )
        for method_id, value in historical.items():
            metrics = value["metrics"]
            methods[method_id] = {
                "success": value["success"],
                "source": value["source"],
                "objective": value["objective"],
                "gap_to_pool_best": value["gap_to_pool_best"],
                "pre_fallback_success": bool(metrics["pre_fallback_success"]),
                "pre_fallback_gap": metrics["best_pre_fallback_gap_to_pool_best"],
                "raw_any_feasible": bool(metrics["raw_any_feasible"]),
                "raw_feasible_rate": metrics["raw_feasible_rate"],
                "fallback_invoked": bool(metrics["fallback_invoked"]),
                "sampling_seconds": float(metrics["sampling_seconds"]),
                "total_seconds": float(metrics["total_seconds"]),
                "selected_source": value["source"],
            }
        write_json(
            path,
            {
                "schema_version": "1.0",
                "scope": CALIBRATION_SCOPE,
                "partition": lock["checkpoint_partition"],
                "instance_id": item.instance.instance_id,
                "pool_best": pool_best,
                "lock": _relative(root, lock_path),
                "lock_sha256": file_sha256(lock_path),
                "methods": methods,
            },
        )
        completed += 1
    return {
        "smoke": smoke,
        "instances": count,
        "completed": completed,
        "output_root": _relative(root, output_root),
    }


def _aggregate_method(records: Iterable[Mapping[str, Any]], method_id: str) -> dict[str, Any]:
    entries = [record["methods"][method_id] for record in records]
    pre = [entry for entry in entries if entry["pre_fallback_success"]]
    return {
        "records": len(entries),
        "final_success_rate": mean(bool(entry["success"]) for entry in entries),
        "mean_gap_to_pool_best": mean(float(entry["gap_to_pool_best"]) for entry in entries if entry["gap_to_pool_best"] is not None),
        "pre_fallback_success_rate": len(pre) / len(entries),
        "mean_pre_fallback_gap": None if not pre else mean(float(entry["pre_fallback_gap"]) for entry in pre),
        "raw_any_feasibility": mean(bool(entry["raw_any_feasible"]) for entry in entries),
        "mean_sampling_seconds": mean(float(entry["sampling_seconds"]) for entry in entries),
        "mean_total_seconds": mean(float(entry["total_seconds"]) for entry in entries),
        "fallback_invocation_rate": mean(bool(entry["fallback_invoked"]) for entry in entries),
    }


def finalize_stage3_calibration(
    lock_path: str | Path, *, implementation_root: str | Path
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_stage3_preparation_lock(lock_path, implementation_root=root)
    output_root = _resolve(root, lock["calibration_root"])
    records = [
        _read_json(output_root / "records" / f"{instance_id}.json")
        for instance_id in lock["checkpoint_instance_ids"]
    ]
    if any(record.get("scope") != CALIBRATION_SCOPE for record in records):
        raise ValueError("Invalid Stage 3 calibration record scope.")
    method_ids = sorted(records[0]["methods"])
    if any(set(record["methods"]) != set(method_ids) for record in records):
        raise ValueError("Stage 3 calibration method sets disagree.")
    aggregate = {
        method_id: _aggregate_method(records, method_id) for method_id in method_ids
    }
    deterministic = aggregate["masked_deterministic_k1"]
    direct_ids = [method_id for method_id in method_ids if method_id.startswith("direct_k")]
    selected_direct = min(
        direct_ids,
        key=lambda method_id: abs(
            aggregate[method_id]["mean_total_seconds"]
            / deterministic["mean_total_seconds"]
            - 1.0
        ),
    )
    direct = aggregate[selected_direct]
    stochastic_id = f"masked_stochastic_k{int(lock['calibration']['stochastic_masked_samples'])}"
    stochastic = aggregate[stochastic_id]
    timing = {
        "selected_direct_method": selected_direct,
        "masked_to_direct_total_time_ratio": deterministic["mean_total_seconds"]
        / direct["mean_total_seconds"],
        "stochastic_to_deterministic_total_time_ratio": stochastic["mean_total_seconds"]
        / deterministic["mean_total_seconds"],
        "masked_to_direct_sampling_time_ratio": deterministic["mean_sampling_seconds"]
        / direct["mean_sampling_seconds"],
        "stochastic_to_deterministic_sampling_time_ratio": stochastic["mean_sampling_seconds"]
        / deterministic["mean_sampling_seconds"],
    }
    timing_checks = {
        "partial_conditioning": timing["masked_to_direct_total_time_ratio"]
        <= float(lock["gates"]["partial_conditioning"]["maximum_time_ratio_to_direct"]),
        "diffusion_specific": timing["stochastic_to_deterministic_total_time_ratio"]
        <= float(lock["gates"]["diffusion_specific"]["maximum_time_ratio_to_deterministic"]),
    }
    evidence = {
        "schema_version": "1.0",
        "scope": CALIBRATION_SCOPE,
        "lock": _relative(root, lock_path),
        "lock_sha256": file_sha256(lock_path),
        "partition": lock["checkpoint_partition"],
        "records": len(records),
        "aggregate": aggregate,
        "timing": timing,
        "timing_checks": timing_checks,
        # R3-C controls multi-seed continuation after pilot; it does not block
        # the pilot evidence needed to distinguish Outcome A from Outcome B.
        "pilot_authorized": timing_checks["partial_conditioning"],
        "multi_seed_timing_precheck": timing_checks["diffusion_specific"],
        "record_sha256": {
            record["instance_id"]: file_sha256(
                output_root / "records" / f"{record['instance_id']}.json"
            )
            for record in records
        },
    }
    evidence_path = output_root / "calibration_evidence.json"
    write_json(evidence_path, evidence)
    contract = {
        "schema_version": "1.0",
        "scope": PILOT_CONTRACT_SCOPE,
        "preparation_lock": _relative(root, lock_path),
        "preparation_lock_sha256": file_sha256(lock_path),
        "calibration_evidence": _relative(root, evidence_path),
        "calibration_evidence_sha256": file_sha256(evidence_path),
        "pilot_authorized": evidence["pilot_authorized"],
        "selected_direct_method": selected_direct,
        "methods": [
            selected_direct,
            "masked_deterministic_k1",
            stochastic_id,
            "diffusion_final_k4",
            "rescue_all_five_b12",
        ],
        "gates": lock["gates"],
        "pilot_partition": lock["pilot_partition"],
        "pilot_instance_ids": lock["pilot_instance_ids"],
        "pilot_root": lock["pilot_root"],
    }
    contract_path = _resolve(root, lock["pilot_contract_lock"])
    write_json(contract_path, contract)
    return {
        "evidence": _relative(root, evidence_path),
        "contract": _relative(root, contract_path),
        "selected_direct_method": selected_direct,
        "timing": timing,
        "timing_checks": timing_checks,
        "pilot_authorized": evidence["pilot_authorized"],
        "contract_sha256": file_sha256(contract_path),
    }
