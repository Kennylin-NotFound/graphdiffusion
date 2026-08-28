"""Locked and resumable Phase 6E-A inference-only ablation campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.training import audit_dataset_freeze

from .aggregation import aggregate_run_directories
from .evaluation import evaluate_experiment
from .schema import file_sha256, load_experiment_manifest, manifest_from_mapping
from .training_aggregation import verify_checkpoint_freeze

PHASE6E_A_SCOPE = "phase6e_a_locked_inference_ablation"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _checkpoint_map(freeze: dict[str, Any]) -> dict[int, Path]:
    checkpoints = {}
    for run in freeze["runs"]:
        seed = int(run["seed"])
        path = Path(run["best_checkpoint"]).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if str(payload.get("model_kind", "diffusion")) != "diffusion":
            raise ValueError("Phase 6E-A requires diffusion checkpoints.")
        checkpoints[seed] = path
    return checkpoints


def _method(
    root: Path,
    campaign: dict[str, Any],
    checkpoint: Path,
    method_id: str,
    kind: str,
    **overrides: Any,
) -> dict[str, Any]:
    inference = dict(campaign["baseline"])
    inference.update(overrides)
    return {
        "method_id": method_id,
        "kind": kind,
        "checkpoint": _relative(root, checkpoint),
        "proposal_group": str(campaign["proposal_group"]),
        "inference": inference,
    }


def _group_methods(
    root: Path,
    campaign: dict[str, Any],
    checkpoint: Path,
    group: str,
) -> list[dict[str, Any]]:
    grid = campaign["grids"][group]
    if group == "postprocessing":
        kinds = {
            "raw_only": "learned_raw_only",
            "repair": "learned_repair",
            "hybrid": "learned_hybrid",
        }
        return [
            _method(root, campaign, checkpoint, f"post_{value}", kinds[str(value)])
            for value in grid
        ]
    if group == "reverse_steps":
        return [
            _method(
                root,
                campaign,
                checkpoint,
                f"steps_{int(value)}",
                "learned_hybrid",
                reverse_steps=int(value),
            )
            for value in grid
        ]
    if group == "proposal_count":
        batch_size = int(campaign["baseline"]["sample_batch_size"])
        return [
            _method(
                root,
                campaign,
                checkpoint,
                f"proposals_{int(value)}",
                "learned_hybrid",
                num_samples=int(value),
                sample_batch_size=min(batch_size, int(value)),
            )
            for value in grid
        ]
    if group == "repair_max_moves":
        return [
            _method(
                root,
                campaign,
                checkpoint,
                f"repair_moves_{int(value)}",
                "learned_hybrid",
                repair_max_moves=int(value),
            )
            for value in grid
        ]
    raise ValueError(f"Unsupported Phase 6E-A group: {group}")


def _manifest(
    root: Path,
    campaign: dict[str, Any],
    checkpoint: Path,
    group: str,
    seed: int,
) -> dict[str, Any]:
    methods = _group_methods(root, campaign, checkpoint, group)
    method_ids = [method["method_id"] for method in methods]
    return {
        "experiment": {
            "schema_version": "1.0",
            "name": f"phase6e-a-{group}-seed{seed}",
            "dataset_root": str(campaign["dataset_root"]),
            "dataset_freeze": str(campaign["dataset_freeze"]),
            "partitions": list(campaign["partitions"]),
            "seed": seed,
            "device": str(campaign["device"]),
            "deterministic": bool(campaign["deterministic"]),
            "output_root": f"{campaign['output_root']}/{group}",
            "methods": methods,
            "claims": [
                {
                    "claim_id": f"phase6e_a_{group}",
                    "question": f"How does {group} affect verified inference?",
                    "hypothesis": "The fixed variants expose a quality, feasibility, and runtime tradeoff.",
                    "comparison": method_ids,
                    "primary_metric": "gap_to_pool_best",
                }
            ],
        }
    }


def prepare_phase6e_a_campaign(
    config_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Generate, validate, hash, and lock all Phase 6E-A final-ID manifests."""

    root = Path(implementation_root).resolve()
    campaign = load_config(config_path)["campaign"]
    if str(campaign["schema_version"]) != "1.0":
        raise ValueError("Unsupported Phase 6E-A campaign schema.")
    seeds = tuple(int(seed) for seed in campaign["seeds"])
    freeze_path = _resolve(root, campaign["checkpoint_freeze"])
    freeze = verify_checkpoint_freeze(freeze_path)
    checkpoints = _checkpoint_map(freeze)
    if set(seeds) != set(checkpoints):
        raise ValueError("Campaign seeds and checkpoint-freeze seeds disagree.")
    if {"train", "validation"} & set(campaign["partitions"]):
        raise ValueError("Phase 6E-A final campaign must contain test partitions only.")

    dataset_root = _resolve(root, campaign["dataset_root"])
    dataset_freeze = dataset_root / str(campaign["dataset_freeze"])
    audit_dataset_freeze(dataset_root)
    manifest_root = _resolve(root, campaign["generated_manifest_root"])
    manifest_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for group in campaign["grids"]:
        for seed in seeds:
            payload = _manifest(root, campaign, checkpoints[seed], group, seed)
            manifest_from_mapping(payload["experiment"])
            path = manifest_root / f"{payload['experiment']['name']}.yaml"
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            entries.append(
                {
                    "group": group,
                    "seed": seed,
                    "path": _relative(root, path),
                    "sha256": file_sha256(path),
                }
            )
    lock = {
        "schema_version": "1.0",
        "scope": PHASE6E_A_SCOPE,
        "name": str(campaign["name"]),
        "config_path": _relative(root, Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        "checkpoint_freeze": _relative(root, freeze_path),
        "checkpoint_freeze_sha256": file_sha256(freeze_path),
        "dataset_root": _relative(root, dataset_root),
        "dataset_freeze": _relative(root, dataset_freeze),
        "dataset_freeze_sha256": file_sha256(dataset_freeze),
        "seeds": list(seeds),
        "contract": {
            "partitions": list(campaign["partitions"]),
            "device": campaign["device"],
            "deterministic": campaign["deterministic"],
            "proposal_group": campaign["proposal_group"],
            "baseline": campaign["baseline"],
            "grids": campaign["grids"],
        },
        "manifests": entries,
    }
    lock_path = _resolve(root, campaign["lock_path"])
    write_json(lock_path, lock)
    return verify_phase6e_a_lock(lock_path, implementation_root=root)


def verify_phase6e_a_lock(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Verify every input and generated manifest in a Phase 6E-A lock."""

    root = Path(implementation_root).resolve()
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("scope") != PHASE6E_A_SCOPE:
        raise ValueError("Unsupported Phase 6E-A lock scope.")
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("checkpoint_freeze", "checkpoint_freeze_sha256"),
        ("dataset_freeze", "dataset_freeze_sha256"),
    ):
        path = _resolve(root, lock[path_key])
        if file_sha256(path) != lock[hash_key]:
            raise ValueError(f"Phase 6E-A lock hash mismatch: {path}")
    verify_checkpoint_freeze(_resolve(root, lock["checkpoint_freeze"]))
    audit_dataset_freeze(_resolve(root, lock["dataset_root"]))
    for entry in lock["manifests"]:
        path = _resolve(root, entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Phase 6E-A manifest hash mismatch: {path}")
        manifest = load_experiment_manifest(path)
        if {"train", "validation"} & set(manifest.partitions):
            raise ValueError("Phase 6E-A lock contains a non-test partition.")
    return lock


def run_phase6e_a_campaign(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
    groups: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run selected locked groups and preserve a resumable artifact index."""

    root = Path(implementation_root).resolve()
    lock = verify_phase6e_a_lock(lock_path, implementation_root=root)
    requested = None if groups is None else set(groups)
    valid = {entry["group"] for entry in lock["manifests"]}
    if requested is not None and not requested <= valid:
        raise ValueError(f"Unknown Phase 6E-A groups: {sorted(requested - valid)}")
    campaign = load_config(_resolve(root, lock["config_path"]))["campaign"]
    output_root = _resolve(root, campaign["output_root"])
    index_path = output_root / "run_index.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.exists()
        else {"schema_version": "1.0", "scope": PHASE6E_A_SCOPE, "runs": {}}
    )
    for entry in lock["manifests"]:
        if requested is not None and entry["group"] not in requested:
            continue
        key = entry["path"]
        existing = index["runs"].get(key)
        if existing is not None:
            run = Path(existing["run_directory"])
            if all(
                (run / filename).exists()
                and file_sha256(run / filename) == existing[hash_key]
                for filename, hash_key in (
                    ("summary.json", "summary_sha256"),
                    ("resolved_manifest.json", "resolved_manifest_sha256"),
                    ("records.jsonl", "records_sha256"),
                )
            ):
                continue
        run = evaluate_experiment(
            load_experiment_manifest(_resolve(root, entry["path"])),
            implementation_root=root,
        )
        index["runs"][key] = {
            **entry,
            "run_directory": str(run.resolve()),
            "summary_sha256": file_sha256(run / "summary.json"),
            "resolved_manifest_sha256": file_sha256(run / "resolved_manifest.json"),
            "records_sha256": file_sha256(run / "records.jsonl"),
        }
        write_json(index_path, index)
    return index


def finalize_phase6e_a_campaign(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Verify completed runs, aggregate each group, and freeze Phase 6E-A."""

    root = Path(implementation_root).resolve()
    lock = verify_phase6e_a_lock(lock_path, implementation_root=root)
    campaign = load_config(_resolve(root, lock["config_path"]))["campaign"]
    output_root = _resolve(root, campaign["output_root"])
    index_path = output_root / "run_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if set(index["runs"]) != {entry["path"] for entry in lock["manifests"]}:
        raise ValueError("Phase 6E-A campaign is incomplete.")
    for record in index["runs"].values():
        run = Path(record["run_directory"])
        for filename, hash_key in (
            ("summary.json", "summary_sha256"),
            ("resolved_manifest.json", "resolved_manifest_sha256"),
            ("records.jsonl", "records_sha256"),
        ):
            if file_sha256(run / filename) != record[hash_key]:
                raise ValueError(f"Phase 6E-A run hash mismatch: {run / filename}")
    aggregates = {}
    for group in campaign["grids"]:
        runs = [
            Path(record["run_directory"])
            for record in index["runs"].values()
            if record["group"] == group
        ]
        path = output_root / f"{group}_five_seed.json"
        aggregate_run_directories(runs, output=path)
        aggregates[group] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    evidence = {
        "schema_version": "1.0",
        "scope": "phase6e_a_final_evidence",
        "campaign_lock": str(Path(lock_path).resolve()),
        "campaign_lock_sha256": file_sha256(lock_path),
        "run_index": str(index_path.resolve()),
        "run_index_sha256": file_sha256(index_path),
        "aggregates": aggregates,
        "runs": list(index["runs"].values()),
    }
    write_json(output_root / "final_evidence_freeze.json", evidence)
    return evidence
