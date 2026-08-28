"""Sealed final-ID evaluation for Phase 6E-E trajectory rescue."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze

from .phase6ed_campaign import verify_phase6ed_final_evidence
from .phase6ee_stage2a import (
    PHASE6EE_STAGE2A_EVIDENCE_SCOPE,
    aggregate_phase6ee_stage2a_records,
    run_direct_method,
    run_trajectory_diffusion_methods,
    verify_phase6ee_stage2a_calibration,
    verify_phase6ee_stage2a_lock,
)
from .runtime import load_learned_solver
from .schema import file_sha256
from .training_aggregation import verify_checkpoint_freeze

PHASE6EE_STAGE2B_SCOPE = "phase6e_e_stage2b_sealed_final_id"
PHASE6EE_STAGE2B_EVIDENCE_SCOPE = "phase6e_e_stage2b_final_evidence"

_QUALITY_METRIC_KEYS = (
    "best_pre_fallback_gap_to_pool_best",
    "best_pre_fallback_objective",
    "best_pre_fallback_source",
    "best_raw_gap_to_pool_best",
    "best_raw_objective",
    "fallback_search_nodes",
    "fallback_success",
    "final_success",
    "num_raw_proposals",
    "pre_fallback_success",
    "raw_any_feasible",
    "raw_capacity_violation_count",
    "raw_capacity_violation_rate",
    "raw_feasible_count",
    "raw_feasible_rate",
    "raw_link_violation_count",
    "raw_link_violation_rate",
    "raw_pairwise_hamming",
    "raw_unique_count",
    "raw_unique_rate",
    "repair_attempts",
    "repair_success_rate",
    "repair_successes",
    "total_repair_moves",
)


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_map(
    freeze: Mapping[str, Any],
    root: Path,
    seeds: Iterable[int],
    *,
    expected_kind: str,
) -> dict[int, dict[str, Any]]:
    expected = {int(seed) for seed in seeds}
    result: dict[int, dict[str, Any]] = {}
    for run in freeze["runs"]:
        seed = int(run["seed"])
        if seed not in expected:
            continue
        checkpoint = Path(run["best_checkpoint"]).resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        kind = str(payload.get("model_kind", "diffusion"))
        if kind != expected_kind:
            raise ValueError(f"Expected {expected_kind} checkpoint: {checkpoint}")
        result[seed] = {
            "seed": seed,
            "checkpoint": _relative(root, checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
        }
    if set(result) != expected:
        raise ValueError(f"{expected_kind} checkpoints do not match Stage 2B seeds.")
    return result


def _verify_stage2a_evidence(
    evidence_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence = _read_json(evidence_path)
    if evidence.get("scope") != PHASE6EE_STAGE2A_EVIDENCE_SCOPE:
        raise ValueError("Unsupported Stage 2A evidence scope.")
    if evidence.get("partition") != "validation" or evidence.get("split") != "confirmation":
        raise ValueError("Stage 2A evidence is not the untouched confirmation split.")
    if not evidence.get("gate_r2", {}).get("passed"):
        raise ValueError("Stage 2A Gate R2 did not authorize final-ID evaluation.")
    if evidence["gate_r2"].get("recommendation") != "sealed_final_id_rescue_authorized":
        raise ValueError("Stage 2A recommendation does not authorize Stage 2B.")
    lock_path = _resolve(root, evidence["lock"])
    if file_sha256(lock_path) != evidence["lock_sha256"]:
        raise ValueError("Stage 2A lock changed after confirmation.")
    lock = verify_phase6ee_stage2a_lock(lock_path, implementation_root=root)
    calibration_path = _resolve(root, evidence["calibration_freeze"])
    if file_sha256(calibration_path) != evidence["calibration_freeze_sha256"]:
        raise ValueError("Stage 2A calibration freeze changed after confirmation.")
    calibration = verify_phase6ee_stage2a_calibration(
        calibration_path,
        lock_path=lock_path,
        implementation_root=root,
    )
    if calibration["selected_variant"] != evidence["gate_r2"]["selected_variant"]:
        raise ValueError("Stage 2A calibration and confirmation selections disagree.")
    for relative, expected in evidence["record_hashes"].items():
        if file_sha256(_resolve(root, relative)) != expected:
            raise ValueError(f"Stage 2A confirmation record changed: {relative}")
    return evidence, lock, calibration


def _normal_method_id(method_id: str) -> str:
    return "diffusion_final_k4" if method_id == "diffusion_k4" else method_id


def quality_payload(
    entry: Mapping[str, Any],
    *,
    method_seed: int,
    pool_best: float,
) -> dict[str, Any]:
    """Return timing-free solver quality used for exact reference replay."""

    metrics = entry["metrics"]
    return {
        "method_seed": int(method_seed),
        "pool_best": float(pool_best),
        "success": bool(entry["success"]),
        "source": str(entry["source"]),
        "objective": None if entry["objective"] is None else float(entry["objective"]),
        "gap_to_pool_best": (
            None
            if entry["gap_to_pool_best"] is None
            else float(entry["gap_to_pool_best"])
        ),
        "metrics": {key: metrics.get(key) for key in _QUALITY_METRIC_KEYS},
    }


def quality_fingerprint(
    references: Mapping[tuple[int, str, str], Mapping[str, Any]],
    *,
    method_id: str,
) -> str:
    selected = [
        {
            "seed": seed,
            "instance_id": instance_id,
            "method_id": current_method,
            "quality": value,
        }
        for (seed, instance_id, current_method), value in sorted(references.items())
        if current_method == method_id
    ]
    if not selected:
        raise ValueError(f"No quality records for {method_id}.")
    return _canonical_sha256(selected)


def _phase6ed_reference_records(
    evidence: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[str, str]]:
    references: dict[tuple[int, str, str], dict[str, Any]] = {}
    namespaces: dict[str, str] = {}
    for run in evidence["runs"]:
        run_directory = Path(run["run_directory"]).resolve()
        records_path = run_directory / "records.jsonl"
        resolved_path = run_directory / "resolved_manifest.json"
        if file_sha256(records_path) != run["records_sha256"]:
            raise ValueError(f"Phase 6E-D records changed: {records_path}")
        if file_sha256(resolved_path) != run["resolved_manifest_sha256"]:
            raise ValueError(f"Phase 6E-D manifest changed: {resolved_path}")
        resolved = _read_json(resolved_path)
        seed = int(resolved["manifest"]["seed"])
        for method in resolved["manifest"]["methods"]:
            method_id = str(method["method_id"])
            if method_id in {"diffusion_k4", "direct_k96"}:
                namespaces[_normal_method_id(method_id)] = str(method["proposal_group"])
        for record in _read_jsonl(records_path):
            method_id = _normal_method_id(str(record["method_id"]))
            if method_id not in {"diffusion_final_k4", "direct_k96"}:
                continue
            key = (seed, str(record["instance_id"]), method_id)
            references[key] = quality_payload(
                record,
                method_seed=int(record["method_seed"]),
                pool_best=float(record["pool_best"]),
            )
    return references, namespaces


def _variant_from_stage2a(
    stage2a_lock: Mapping[str, Any], selected_variant: str
) -> dict[str, Any]:
    matches = [
        dict(variant)
        for variant in stage2a_lock["variants"]
        if variant["method_id"] == selected_variant
    ]
    if len(matches) != 1:
        raise ValueError("Stage 2A selected variant is missing or ambiguous.")
    return matches[0]


def prepare_phase6ee_stage2b(
    config_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Create the immutable lock before any Stage 2B final-ID inference."""

    root = Path(implementation_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)["stage2b"]
    if str(config["schema_version"]) != "1.0":
        raise ValueError("Unsupported Phase 6E-E Stage 2B schema.")
    partition = str(config["partition"])
    forbidden = {str(value) for value in config["forbidden_partitions"]}
    if partition != "test_id" or partition in forbidden:
        raise ValueError("Stage 2B is restricted to the sealed test_id partition.")
    seeds = tuple(int(seed) for seed in config["seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Stage 2B seeds must be unique and nonempty.")

    stage2a_path = _resolve(root, config["stage2a_evidence"])
    stage2a_evidence, stage2a_lock, calibration = _verify_stage2a_evidence(
        stage2a_path, root=root
    )
    selected_variant = str(config["selected_variant"])
    if selected_variant != stage2a_evidence["gate_r2"]["selected_variant"]:
        raise ValueError("Stage 2B attempted to change the Stage 2A selection.")
    selected_spec = _variant_from_stage2a(stage2a_lock, selected_variant)

    prior_path = _resolve(root, config["phase6ed_final_evidence"])
    prior = verify_phase6ed_final_evidence(prior_path, implementation_root=root)
    prior_lock = _read_json(_resolve(root, prior["final_lock"]))
    if prior_lock["partition"] != partition or tuple(prior_lock["seeds"]) != seeds:
        raise ValueError("Stage 2B and Phase 6E-D final contracts disagree.")
    references, namespaces = _phase6ed_reference_records(prior, root=root)
    expected_namespaces = {
        "diffusion_final_k4": str(config["proposal_seed_namespaces"]["diffusion"]),
        "direct_k96": str(config["proposal_seed_namespaces"]["direct"]),
    }
    if namespaces != expected_namespaces:
        raise ValueError("Stage 2B proposal namespaces do not reproduce Phase 6E-D.")

    dataset_root = _resolve(root, config["dataset_root"])
    audit_dataset_freeze(dataset_root)
    dataset_freeze = dataset_root / str(config["dataset_freeze"])
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=(partition,),
        require_freeze=True,
    )
    instance_ids = tuple(
        dataset[index].instance.instance_id for index in range(len(dataset))
    )
    if len(instance_ids) != 128 or len(set(instance_ids)) != len(instance_ids):
        raise ValueError("Stage 2B expects exactly 128 unique final-ID instances.")

    diffusion_freeze_path = _resolve(root, config["diffusion_checkpoint_freeze"])
    direct_freeze_path = _resolve(root, config["direct_checkpoint_freeze"])
    diffusion_freeze = verify_checkpoint_freeze(diffusion_freeze_path)
    direct_freeze = verify_checkpoint_freeze(direct_freeze_path)
    diffusion = _checkpoint_map(
        diffusion_freeze, root, seeds, expected_kind="diffusion"
    )
    direct = _checkpoint_map(direct_freeze, root, seeds, expected_kind="direct")

    output_root = _resolve(root, config["output_root"])
    lock = {
        "schema_version": "1.0",
        "scope": PHASE6EE_STAGE2B_SCOPE,
        "config_path": _relative(root, config_path),
        "config_sha256": file_sha256(config_path),
        "stage2a_evidence": _relative(root, stage2a_path),
        "stage2a_evidence_sha256": file_sha256(stage2a_path),
        "stage2a_lock_sha256": stage2a_evidence["lock_sha256"],
        "stage2a_calibration_sha256": stage2a_evidence[
            "calibration_freeze_sha256"
        ],
        "phase6ed_final_evidence": _relative(root, prior_path),
        "phase6ed_final_evidence_sha256": file_sha256(prior_path),
        "phase6ed_final_lock_sha256": prior["final_lock_sha256"],
        "dataset_root": _relative(root, dataset_root),
        "dataset_freeze": _relative(root, dataset_freeze),
        "dataset_freeze_sha256": file_sha256(dataset_freeze),
        "diffusion_checkpoint_freeze": _relative(root, diffusion_freeze_path),
        "diffusion_checkpoint_freeze_sha256": file_sha256(diffusion_freeze_path),
        "direct_checkpoint_freeze": _relative(root, direct_freeze_path),
        "direct_checkpoint_freeze_sha256": file_sha256(direct_freeze_path),
        "partition": partition,
        "forbidden_partitions": sorted(forbidden),
        "seeds": list(seeds),
        "instance_ids": list(instance_ids),
        "expected_record_count": len(seeds) * len(instance_ids),
        "diffusion_checkpoints": [diffusion[seed] for seed in seeds],
        "direct_checkpoints": [direct[seed] for seed in seeds],
        "selected_variant": selected_variant,
        "variants": [
            {
                "method_id": "diffusion_final_k4",
                "anchor_set": "final_only",
                "anchor_indices": [],
                "repair_candidate_limit": None,
            },
            selected_spec,
        ],
        "methods": ["diffusion_final_k4", selected_variant, "direct_k96"],
        "diffusion": dict(config["diffusion"]),
        "direct": dict(config["direct"]),
        "postprocessing": dict(config["postprocessing"]),
        "proposal_seed_namespaces": {
            "diffusion": expected_namespaces["diffusion_final_k4"],
            "direct": expected_namespaces["direct_k96"],
        },
        "reference_quality_fingerprints": {
            method_id: quality_fingerprint(references, method_id=method_id)
            for method_id in ("diffusion_final_k4", "direct_k96")
        },
        "device": str(config["device"]),
        "deterministic": bool(config["deterministic"]),
        "output_root": _relative(root, output_root),
    }
    lock_path = _resolve(root, config["lock_path"])
    if (output_root / "execution_state.json").exists():
        raise ValueError("Stage 2B has already opened final ID; refusing to relock.")
    write_json(lock_path, lock)
    return verify_phase6ee_stage2b_lock(lock_path, implementation_root=root)


def verify_phase6ee_stage2b_lock(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PHASE6EE_STAGE2B_SCOPE:
        raise ValueError("Unsupported Stage 2B lock scope.")
    if lock["partition"] != "test_id" or lock["partition"] in lock["forbidden_partitions"]:
        raise ValueError("Stage 2B lock escaped the final-ID boundary.")
    if len(lock["seeds"]) != 5 or len(lock["instance_ids"]) != 128:
        raise ValueError("Stage 2B lock must contain five seeds and 128 instances.")
    if int(lock["expected_record_count"]) != 640:
        raise ValueError("Stage 2B lock must contain 640 seed-instance records.")
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("stage2a_evidence", "stage2a_evidence_sha256"),
        ("phase6ed_final_evidence", "phase6ed_final_evidence_sha256"),
        ("dataset_freeze", "dataset_freeze_sha256"),
        ("diffusion_checkpoint_freeze", "diffusion_checkpoint_freeze_sha256"),
        ("direct_checkpoint_freeze", "direct_checkpoint_freeze_sha256"),
    ):
        path = _resolve(root, lock[path_key])
        if file_sha256(path) != lock[hash_key]:
            raise ValueError(f"Stage 2B lock hash mismatch: {path}")
    stage2a, stage2a_lock, calibration = _verify_stage2a_evidence(
        _resolve(root, lock["stage2a_evidence"]), root=root
    )
    if stage2a["lock_sha256"] != lock["stage2a_lock_sha256"]:
        raise ValueError("Stage 2A lock fingerprint changed.")
    if stage2a["calibration_freeze_sha256"] != lock["stage2a_calibration_sha256"]:
        raise ValueError("Stage 2A calibration fingerprint changed.")
    if calibration["selected_variant"] != lock["selected_variant"]:
        raise ValueError("Stage 2B selected variant changed after locking.")
    selected_spec = _variant_from_stage2a(stage2a_lock, lock["selected_variant"])
    if selected_spec != lock["variants"][1]:
        raise ValueError("Stage 2B selected variant parameters changed.")

    prior = verify_phase6ed_final_evidence(
        _resolve(root, lock["phase6ed_final_evidence"]), implementation_root=root
    )
    if prior["final_lock_sha256"] != lock["phase6ed_final_lock_sha256"]:
        raise ValueError("Phase 6E-D final lock fingerprint changed.")
    references, namespaces = _phase6ed_reference_records(prior, root=root)
    expected_namespaces = {
        "diffusion_final_k4": lock["proposal_seed_namespaces"]["diffusion"],
        "direct_k96": lock["proposal_seed_namespaces"]["direct"],
    }
    if namespaces != expected_namespaces:
        raise ValueError("Phase 6E-D proposal namespaces changed.")
    for method_id, expected in lock["reference_quality_fingerprints"].items():
        if quality_fingerprint(references, method_id=method_id) != expected:
            raise ValueError(f"Phase 6E-D quality fingerprint changed: {method_id}")

    dataset_root = _resolve(root, lock["dataset_root"])
    audit_dataset_freeze(dataset_root)
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=(lock["partition"],),
        require_freeze=True,
    )
    current_ids = [dataset[index].instance.instance_id for index in range(len(dataset))]
    if current_ids != lock["instance_ids"]:
        raise ValueError("Stage 2B final-ID instance order changed.")
    verify_checkpoint_freeze(_resolve(root, lock["diffusion_checkpoint_freeze"]))
    verify_checkpoint_freeze(_resolve(root, lock["direct_checkpoint_freeze"]))
    for group in ("diffusion_checkpoints", "direct_checkpoints"):
        for entry in lock[group]:
            path = _resolve(root, entry["checkpoint"])
            if file_sha256(path) != entry["checkpoint_sha256"]:
                raise ValueError(f"Stage 2B checkpoint hash mismatch: {path}")
    return lock


def _record_path(output_root: Path, seed: int, instance_id: str) -> Path:
    return output_root / "records" / f"seed{seed}" / f"{instance_id}.json"


def _reference_key(seed: int, instance_id: str, method_id: str) -> tuple[int, str, str]:
    return int(seed), str(instance_id), str(method_id)


def _record_quality_replays_reference(
    record: Mapping[str, Any],
    references: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> bool:
    seed = int(record["seed"])
    instance_id = str(record["instance_id"])
    pool_best = float(record["pool_best"])
    for method_id, seed_key in (
        ("diffusion_final_k4", "diffusion_seed"),
        ("direct_k96", "direct_seed"),
    ):
        actual = quality_payload(
            record["methods"][method_id],
            method_seed=int(record[seed_key]),
            pool_best=pool_best,
        )
        expected = references.get(_reference_key(seed, instance_id, method_id))
        if actual != expected:
            return False
    return True


def _valid_existing_record(
    record: Mapping[str, Any],
    *,
    lock_sha256: str,
    seed: int,
    instance_id: str,
    diffusion_hash: str,
    direct_hash: str,
    expected_methods: set[str],
    references: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> bool:
    return bool(
        record.get("schema_version") == "1.0"
        and record.get("scope") == PHASE6EE_STAGE2B_SCOPE
        and record.get("lock_sha256") == lock_sha256
        and record.get("partition") == "test_id"
        and int(record.get("seed", -1)) == seed
        and record.get("instance_id") == instance_id
        and record.get("diffusion_checkpoint_sha256") == diffusion_hash
        and record.get("direct_checkpoint_sha256") == direct_hash
        and set(record.get("methods", {})) == expected_methods
        and _record_quality_replays_reference(record, references)
    )


def run_phase6ee_stage2b(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
    seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Open the locked final ID once and resumably evaluate every record."""

    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_phase6ee_stage2b_lock(lock_path, implementation_root=root)
    lock_hash = file_sha256(lock_path)
    requested = (
        tuple(int(seed) for seed in seeds)
        if seeds is not None
        else tuple(int(seed) for seed in lock["seeds"])
    )
    if not requested or not set(requested) <= set(lock["seeds"]):
        raise ValueError("Requested seed is outside the Stage 2B lock.")
    device = torch.device(lock["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the locked Stage 2B campaign.")

    prior = verify_phase6ed_final_evidence(
        _resolve(root, lock["phase6ed_final_evidence"]), implementation_root=root
    )
    references, _ = _phase6ed_reference_records(prior, root=root)
    output_root = _resolve(root, lock["output_root"])
    state_path = output_root / "execution_state.json"
    if state_path.exists():
        state = _read_json(state_path)
        if state.get("lock_sha256") != lock_hash:
            raise ValueError("Stage 2B execution state references another lock.")
    else:
        state = {
            "schema_version": "1.0",
            "scope": PHASE6EE_STAGE2B_SCOPE,
            "lock": _relative(root, lock_path),
            "lock_sha256": lock_hash,
            "partition": "test_id",
            "first_opened_at_utc": _utc_now(),
            "status": "running",
            "completed_records": 0,
            "expected_records": int(lock["expected_record_count"]),
            "resume_invocations": 0,
        }
    state["resume_invocations"] = int(state.get("resume_invocations", 0)) + 1
    state["last_resumed_at_utc"] = _utc_now()
    state["status"] = "running"
    write_json(state_path, state)

    dataset = LabeledDeploymentDataset(
        _resolve(root, lock["dataset_root"]),
        partitions=(lock["partition"],),
        require_freeze=True,
    )
    items = {
        dataset[index].instance.instance_id: dataset[index]
        for index in range(len(dataset))
    }
    diffusion_entries = {
        int(entry["seed"]): entry for entry in lock["diffusion_checkpoints"]
    }
    direct_entries = {
        int(entry["seed"]): entry for entry in lock["direct_checkpoints"]
    }
    expected_methods = set(lock["methods"])
    completed_this_invocation = 0
    reused_this_invocation = 0
    for seed in requested:
        seed_everything(seed, deterministic=bool(lock["deterministic"]))
        diffusion_entry = diffusion_entries[seed]
        direct_entry = direct_entries[seed]
        diffusion_solver = load_learned_solver(
            _resolve(root, diffusion_entry["checkpoint"]), dataset, device
        )
        direct_solver = load_learned_solver(
            _resolve(root, direct_entry["checkpoint"]), dataset, device
        )
        for instance_id in lock["instance_ids"]:
            path = _record_path(output_root, seed, instance_id)
            if path.exists():
                existing = _read_json(path)
                if _valid_existing_record(
                    existing,
                    lock_sha256=lock_hash,
                    seed=seed,
                    instance_id=instance_id,
                    diffusion_hash=diffusion_entry["checkpoint_sha256"],
                    direct_hash=direct_entry["checkpoint_sha256"],
                    expected_methods=expected_methods,
                    references=references,
                ):
                    reused_this_invocation += 1
                    continue
                raise ValueError(f"Stale or non-reproducible Stage 2B record: {path}")
            item = items[instance_id]
            diffusion_seed = derive_seed(
                seed,
                f"{lock['proposal_seed_namespaces']['diffusion']}:{instance_id}",
            )
            direct_seed = derive_seed(
                seed,
                f"{lock['proposal_seed_namespaces']['direct']}:{instance_id}",
            )
            methods = run_trajectory_diffusion_methods(
                lock,
                diffusion_solver,
                item,
                selected_variant=lock["selected_variant"],
                generator=torch.Generator(device=device).manual_seed(diffusion_seed),
                device=device,
            )
            methods["direct_k96"] = run_direct_method(
                lock,
                direct_solver,
                item,
                generator=torch.Generator(device=device).manual_seed(direct_seed),
                device=device,
            )
            record = {
                "schema_version": "1.0",
                "scope": PHASE6EE_STAGE2B_SCOPE,
                "lock_sha256": lock_hash,
                "partition": "test_id",
                "seed": seed,
                "instance_id": instance_id,
                "diffusion_seed": diffusion_seed,
                "direct_seed": direct_seed,
                "diffusion_checkpoint_sha256": diffusion_entry[
                    "checkpoint_sha256"
                ],
                "direct_checkpoint_sha256": direct_entry["checkpoint_sha256"],
                "pool_best": float(np.min(item.pool.latencies)),
                "methods": methods,
            }
            if not _record_quality_replays_reference(record, references):
                raise ValueError(
                    "Stage 2B baseline/direct quality failed exact Phase 6E-D "
                    f"replay before writing: seed={seed}, instance={instance_id}"
                )
            write_json(path, record)
            completed_this_invocation += 1
            state["completed_records"] = sum(
                1 for _ in (output_root / "records").glob("seed*/*.json")
            )
            state["last_completed_seed"] = seed
            state["last_completed_instance"] = instance_id
            write_json(state_path, state)

    total_completed = sum(
        1 for _ in (output_root / "records").glob("seed*/*.json")
    )
    state["completed_records"] = total_completed
    if total_completed == int(lock["expected_record_count"]):
        state["status"] = "completed"
        state["completed_at_utc"] = _utc_now()
    write_json(state_path, state)
    return {
        "status": state["status"],
        "requested_seeds": list(requested),
        "completed_this_invocation": completed_this_invocation,
        "reused_this_invocation": reused_this_invocation,
        "completed_records": total_completed,
        "expected_records": int(lock["expected_record_count"]),
    }


def _collect_records(
    lock: Mapping[str, Any],
    *,
    root: Path,
    lock_sha256: str,
    references: Mapping[tuple[int, str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    output_root = _resolve(root, lock["output_root"])
    diffusion_entries = {
        int(entry["seed"]): entry for entry in lock["diffusion_checkpoints"]
    }
    direct_entries = {
        int(entry["seed"]): entry for entry in lock["direct_checkpoints"]
    }
    expected_methods = set(lock["methods"])
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for seed in lock["seeds"]:
        for instance_id in lock["instance_ids"]:
            path = _record_path(output_root, int(seed), instance_id)
            if not path.exists():
                raise FileNotFoundError(f"Missing Stage 2B record: {path}")
            record = _read_json(path)
            if not _valid_existing_record(
                record,
                lock_sha256=lock_sha256,
                seed=int(seed),
                instance_id=instance_id,
                diffusion_hash=diffusion_entries[int(seed)]["checkpoint_sha256"],
                direct_hash=direct_entries[int(seed)]["checkpoint_sha256"],
                expected_methods=expected_methods,
                references=references,
            ):
                raise ValueError(f"Invalid Stage 2B record contract: {path}")
            records.append(record)
            hashes[_relative(root, path)] = file_sha256(path)
    return records, hashes


def paired_outcomes(
    records: Iterable[Mapping[str, Any]], left: str, right: str
) -> dict[str, int]:
    counts = {"left_wins": 0, "ties": 0, "right_wins": 0}
    for record in records:
        left_value = float(record["methods"][left]["objective"])
        right_value = float(record["methods"][right]["objective"])
        if left_value < right_value - 1e-12:
            counts["left_wins"] += 1
        elif right_value < left_value - 1e-12:
            counts["right_wins"] += 1
        else:
            counts["ties"] += 1
    return counts


def interpret_stage2b(
    aggregate: Mapping[str, Any], *, selected_variant: str
) -> dict[str, Any]:
    methods = aggregate["methods"]
    baseline = methods["diffusion_final_k4"]
    rescue = methods[selected_variant]
    direct = methods["direct_k96"]
    baseline_gap = float(baseline["gap_to_pool_best"])
    rescue_gap = float(rescue["gap_to_pool_best"])
    direct_gap = float(direct["gap_to_pool_best"])
    denominator = baseline_gap - direct_gap
    closure = None if denominator <= 0 else (baseline_gap - rescue_gap) / denominator
    improves_baseline = rescue_gap < baseline_gap - 1e-12
    beats_direct = rescue_gap < direct_gap - 1e-12
    if beats_direct:
        conclusion = "trajectory_rescue_exceeds_time_matched_direct_on_mean_gap"
    elif improves_baseline:
        conclusion = "trajectory_rescue_improves_diffusion_but_direct_remains_stronger"
    else:
        conclusion = "trajectory_rescue_does_not_reproduce_validation_improvement"
    return {
        "conclusion": conclusion,
        "rescue_improves_final_k4": improves_baseline,
        "rescue_beats_direct_k96": beats_direct,
        "gap_closure": closure,
        "rescue_to_direct_time_ratio": float(rescue["total_seconds"])
        / float(direct["total_seconds"]),
        "raw_any_gain_over_final_k4": float(rescue["raw_any_feasible_rate"])
        - float(baseline["raw_any_feasible_rate"]),
        "selected_fallback_reduction": float(
            baseline["selected_source_rates"]["fallback"]
        )
        - float(rescue["selected_source_rates"]["fallback"]),
    }


def _stage2b_report(evidence: Mapping[str, Any]) -> str:
    aggregate = evidence["aggregate"]["overall"]
    selected = evidence["selected_variant"]
    methods = aggregate["methods"]
    interpretation = evidence["interpretation"]
    lines = [
        "# Phase 6E-E Stage 2B 最终 ID 报告",
        "",
        "本报告来自冻结后的五种子 `test_id` 单次比较。Stage 2B 未重新训练、"
        "未重新使用验证集，也未改变硬验证、修复或 fallback 契约。",
        "",
        "| 方法 | Gap | Raw-any | Pre-fallback gap | Selected fallback | Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method_id in ("diffusion_final_k4", selected, "direct_k96"):
        value = methods[method_id]
        pre_gap = value["best_pre_fallback_gap_to_pool_best"]
        lines.append(
            f"| `{method_id}` | {100 * value['gap_to_pool_best']:.3f}% | "
            f"{100 * value['raw_any_feasible_rate']:.2f}% | "
            f"{'N/A' if pre_gap is None else f'{100 * pre_gap:.3f}%'} | "
            f"{100 * value['selected_source_rates']['fallback']:.2f}% | "
            f"{value['total_seconds']:.4f} |"
        )
    closure = interpretation["gap_closure"]
    lines.extend(
        [
            "",
            "## 审计结论",
            "",
            f"- 记录数：{aggregate['records']}（128 个实例 × 5 个种子）。",
            "- Phase 6E-D 的 final K=4 与 direct K=96 质量指纹逐实例精确复现。",
            f"- 相对 final K=4/direct K=96 差距闭合率："
            f"{'N/A' if closure is None else f'{100 * closure:.2f}%'}。",
            f"- rescue/direct 在线时间比："
            f"{interpretation['rescue_to_direct_time_ratio']:.3f}。",
            f"- 成对结果（final K=4 / tie / rescue）："
            f"{evidence['paired']['final_vs_rescue']['left_wins']} / "
            f"{evidence['paired']['final_vs_rescue']['ties']} / "
            f"{evidence['paired']['final_vs_rescue']['right_wins']}。",
            f"- 成对结果（rescue / tie / direct K=96）："
            f"{evidence['paired']['rescue_vs_direct']['left_wins']} / "
            f"{evidence['paired']['rescue_vs_direct']['ties']} / "
            f"{evidence['paired']['rescue_vs_direct']['right_wins']}。",
            f"- 冻结解释：`{interpretation['conclusion']}`。",
            "",
            "该结论仅适用于当前冻结的合成 ID 契约。是否修改论文中的扩散优势"
            "表述，应以该冻结解释为边界，不得回到验证集继续调参。",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_phase6ee_stage2b(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Aggregate all 640 records and freeze the Stage 2B decision evidence."""

    root = Path(implementation_root).resolve()
    lock_path = Path(lock_path).resolve()
    lock = verify_phase6ee_stage2b_lock(lock_path, implementation_root=root)
    lock_hash = file_sha256(lock_path)
    output_root = _resolve(root, lock["output_root"])
    state_path = output_root / "execution_state.json"
    if not state_path.exists():
        raise FileNotFoundError("Stage 2B final ID has not been opened.")
    state = _read_json(state_path)
    if state.get("lock_sha256") != lock_hash or state.get("status") != "completed":
        raise ValueError("Stage 2B execution is incomplete or references another lock.")

    prior = verify_phase6ed_final_evidence(
        _resolve(root, lock["phase6ed_final_evidence"]), implementation_root=root
    )
    references, _ = _phase6ed_reference_records(prior, root=root)
    records, record_hashes = _collect_records(
        lock,
        root=root,
        lock_sha256=lock_hash,
        references=references,
    )
    if len(records) != int(lock["expected_record_count"]):
        raise ValueError("Stage 2B record count is incomplete.")
    overall = aggregate_phase6ee_stage2a_records(records)
    by_seed = {
        str(seed): aggregate_phase6ee_stage2a_records(
            record for record in records if int(record["seed"]) == int(seed)
        )
        for seed in lock["seeds"]
    }
    new_references: dict[tuple[int, str, str], dict[str, Any]] = {}
    for record in records:
        for method_id, seed_key in (
            ("diffusion_final_k4", "diffusion_seed"),
            ("direct_k96", "direct_seed"),
        ):
            key = _reference_key(record["seed"], record["instance_id"], method_id)
            new_references[key] = quality_payload(
                record["methods"][method_id],
                method_seed=int(record[seed_key]),
                pool_best=float(record["pool_best"]),
            )
    replay_fingerprints = {
        method_id: quality_fingerprint(new_references, method_id=method_id)
        for method_id in ("diffusion_final_k4", "direct_k96")
    }
    if replay_fingerprints != lock["reference_quality_fingerprints"]:
        raise ValueError("Stage 2B did not reproduce Phase 6E-D quality fingerprints.")

    selected = str(lock["selected_variant"])
    paired = {
        "final_vs_rescue": paired_outcomes(
            records, "diffusion_final_k4", selected
        ),
        "rescue_vs_direct": paired_outcomes(records, selected, "direct_k96"),
        "final_vs_direct": paired_outcomes(
            records, "diffusion_final_k4", "direct_k96"
        ),
    }
    interpretation = interpret_stage2b(overall, selected_variant=selected)
    record_index_path = output_root / "record_index.json"
    aggregate_path = output_root / "final_id_five_seed.json"
    report_path = root / "PHASE6E_E_STAGE2B_REPORT_ZH.md"
    write_json(
        record_index_path,
        {
            "schema_version": "1.0",
            "scope": PHASE6EE_STAGE2B_SCOPE,
            "lock_sha256": lock_hash,
            "record_count": len(records),
            "record_hashes": record_hashes,
        },
    )
    write_json(
        aggregate_path,
        {
            "schema_version": "1.0",
            "scope": PHASE6EE_STAGE2B_SCOPE,
            "partition": "test_id",
            "seeds": lock["seeds"],
            "selected_variant": selected,
            "overall": overall,
            "by_seed": by_seed,
            "paired": paired,
            "interpretation": interpretation,
        },
    )
    evidence = {
        "schema_version": "1.0",
        "scope": PHASE6EE_STAGE2B_EVIDENCE_SCOPE,
        "lock": _relative(root, lock_path),
        "lock_sha256": lock_hash,
        "execution_state": _relative(root, state_path),
        "execution_state_sha256": file_sha256(state_path),
        "record_index": _relative(root, record_index_path),
        "record_index_sha256": file_sha256(record_index_path),
        "aggregate": {
            "path": _relative(root, aggregate_path),
            "sha256": file_sha256(aggregate_path),
            "overall": overall,
        },
        "stage2a_evidence_sha256": lock["stage2a_evidence_sha256"],
        "phase6ed_final_evidence_sha256": lock[
            "phase6ed_final_evidence_sha256"
        ],
        "partition": "test_id",
        "seeds": lock["seeds"],
        "record_count": len(records),
        "selected_variant": selected,
        "quality_replay_fingerprints": replay_fingerprints,
        "paired": paired,
        "interpretation": interpretation,
    }
    report_path.write_text(_stage2b_report(evidence), encoding="utf-8")
    evidence["report"] = {
        "path": _relative(root, report_path),
        "sha256": file_sha256(report_path),
    }
    evidence_path = output_root / "final_evidence_freeze.json"
    write_json(evidence_path, evidence)
    return verify_phase6ee_stage2b_evidence(
        evidence_path, implementation_root=root
    )


def verify_phase6ee_stage2b_evidence(
    evidence_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    evidence = _read_json(evidence_path)
    if evidence.get("scope") != PHASE6EE_STAGE2B_EVIDENCE_SCOPE:
        raise ValueError("Unsupported Stage 2B evidence scope.")
    lock_path = _resolve(root, evidence["lock"])
    if file_sha256(lock_path) != evidence["lock_sha256"]:
        raise ValueError("Stage 2B lock changed after evidence freeze.")
    lock = verify_phase6ee_stage2b_lock(lock_path, implementation_root=root)
    for path_key, hash_key in (
        ("execution_state", "execution_state_sha256"),
        ("record_index", "record_index_sha256"),
    ):
        path = _resolve(root, evidence[path_key])
        if file_sha256(path) != evidence[hash_key]:
            raise ValueError(f"Stage 2B evidence artifact changed: {path}")
    for record in (evidence["aggregate"], evidence["report"]):
        path = _resolve(root, record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Stage 2B evidence artifact changed: {path}")
    index = _read_json(_resolve(root, evidence["record_index"]))
    if index["record_count"] != lock["expected_record_count"]:
        raise ValueError("Stage 2B record index has the wrong cardinality.")
    for relative, expected in index["record_hashes"].items():
        if file_sha256(_resolve(root, relative)) != expected:
            raise ValueError(f"Stage 2B record changed after freeze: {relative}")
    if evidence["quality_replay_fingerprints"] != lock[
        "reference_quality_fingerprints"
    ]:
        raise ValueError("Stage 2B evidence lost Phase 6E-D replay parity.")
    return evidence
