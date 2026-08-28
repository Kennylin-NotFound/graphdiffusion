"""Locked Phase 6D-C manifest generation and resumable campaign execution."""

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

SEALED_CAMPAIGN_SCOPE = "phase6d_c_sealed_main_comparisons"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Campaign path must stay inside implementation root: {path}") from error


def _checkpoint_map(freeze: dict[str, Any], expected_kind: str) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for run in freeze["runs"]:
        seed = int(run["seed"])
        path = Path(run["best_checkpoint"]).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        kind = str(payload.get("model_kind", "diffusion"))
        if kind != expected_kind:
            raise ValueError(
                f"Checkpoint {path} has model kind {kind!r}, expected {expected_kind!r}."
            )
        mapping[seed] = path
    return mapping


def _stochastic_methods(
    root: Path,
    campaign: dict[str, Any],
    diffusion_checkpoint: Path,
    direct_checkpoint: Path,
) -> list[dict[str, Any]]:
    shared = {
        "num_samples": int(campaign["proposal_count"]),
        "sample_batch_size": int(campaign["proposal_batch_size"]),
        "repair_max_moves": int(campaign["repair_max_moves"]),
        "fallback_max_search_nodes": int(campaign["fallback_max_search_nodes"]),
    }
    proposal_group = "paired-categorical-proposals"
    return [
        {
            "method_id": "diffusion_hybrid",
            "kind": "learned_hybrid",
            "checkpoint": _relative(root, diffusion_checkpoint),
            "proposal_group": proposal_group,
            "inference": {
                **shared,
                "reverse_steps": int(campaign["diffusion_reverse_steps"]),
            },
        },
        {
            "method_id": "direct_hybrid",
            "kind": "direct_hybrid",
            "checkpoint": _relative(root, direct_checkpoint),
            "proposal_group": proposal_group,
            "inference": dict(shared),
        },
        {
            "method_id": "random_hybrid",
            "kind": "random_hybrid",
            "proposal_group": proposal_group,
            "inference": dict(shared),
        },
        {
            "method_id": "fallback_only",
            "kind": "fallback_only",
            "inference": {
                "fallback_max_search_nodes": int(
                    campaign["fallback_max_search_nodes"]
                )
            },
        },
    ]


def _optimization_methods(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "method_id": "milp_2s",
            "kind": "milp_time_limit",
            "time_limit_seconds": float(campaign["milp_time_limit_seconds"]),
        },
        {"method_id": "greedy_local", "kind": "greedy_local"},
        {
            "method_id": "fallback_only",
            "kind": "fallback_only",
            "inference": {
                "fallback_max_search_nodes": int(
                    campaign["fallback_max_search_nodes"]
                )
            },
        },
    ]


def _claims(group: str) -> list[dict[str, Any]]:
    if group == "stochastic":
        return [
            {
                "claim_id": "diffusion_isolation",
                "question": "Does reverse diffusion improve over one-pass prediction?",
                "hypothesis": "Diffusion improves verified latency under matched graph, labels, proposal count, and post-processing.",
                "comparison": ["diffusion_hybrid", "direct_hybrid"],
                "primary_metric": "gap_to_pool_best",
            },
            {
                "claim_id": "learned_proposal_value",
                "question": "Do learned proposals improve over random proposals and fallback?",
                "hypothesis": "Learned proposals improve verified latency and raw feasibility under the shared hard pipeline.",
                "comparison": [
                    "diffusion_hybrid",
                    "direct_hybrid",
                    "random_hybrid",
                    "fallback_only",
                ],
                "primary_metric": "gap_to_pool_best",
            },
        ]
    return [
        {
            "claim_id": "optimization_baselines",
            "question": "How do deterministic optimization baselines compare under declared online budgets?",
            "hypothesis": "The exact MILP provides strong incumbents while greedy may fail without backtracking.",
            "comparison": ["milp_2s", "greedy_local", "fallback_only"],
            "primary_metric": "success",
        }
    ]


def _manifest_payload(
    *,
    campaign: dict[str, Any],
    dataset_name: str,
    dataset: dict[str, Any],
    group: str,
    seed: int,
    methods: list[dict[str, Any]],
) -> dict[str, Any]:
    name = f"phase6d-c-{dataset_name}-{group}-seed{seed}"
    return {
        "experiment": {
            "schema_version": "1.0",
            "name": name,
            "dataset_root": str(dataset["dataset_root"]),
            "dataset_freeze": str(dataset.get("dataset_freeze", "dataset_freeze.json")),
            "partitions": list(dataset["partitions"]),
            "seed": seed,
            "device": str(campaign["device"]),
            "deterministic": bool(campaign["deterministic"]),
            "output_root": f"{campaign['output_root']}/{dataset_name}-{group}",
            "methods": methods,
            "claims": _claims(group),
        }
    }


def prepare_sealed_campaign(
    config_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Generate, validate, hash, and lock every final-test manifest."""

    root = Path(implementation_root).resolve()
    campaign = load_config(config_path)["campaign"]
    if str(campaign["schema_version"]) != "1.0":
        raise ValueError("Unsupported sealed campaign schema version.")
    seeds = tuple(int(seed) for seed in campaign["seeds"])
    if len(seeds) != len(set(seeds)) or len(seeds) < 2:
        raise ValueError("Sealed campaign requires distinct independent seeds.")

    diffusion_freeze_path = _resolve(root, campaign["diffusion_checkpoint_freeze"])
    direct_freeze_path = _resolve(root, campaign["direct_checkpoint_freeze"])
    diffusion_freeze = verify_checkpoint_freeze(diffusion_freeze_path)
    direct_freeze = verify_checkpoint_freeze(direct_freeze_path)
    diffusion_checkpoints = _checkpoint_map(diffusion_freeze, "diffusion")
    direct_checkpoints = _checkpoint_map(direct_freeze, "direct")
    if set(seeds) != set(diffusion_checkpoints) or set(seeds) != set(direct_checkpoints):
        raise ValueError("Campaign seeds and frozen checkpoint seeds disagree.")

    manifest_root = _resolve(root, campaign["generated_manifest_root"])
    manifest_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    datasets: dict[str, Any] = {}
    forbidden_partitions = {"train", "validation"}
    for dataset_name, dataset in campaign["datasets"].items():
        partitions = tuple(str(value) for value in dataset["partitions"])
        if forbidden_partitions & set(partitions):
            raise ValueError("Sealed campaign must not contain train or validation.")
        dataset_root = _resolve(root, dataset["dataset_root"])
        dataset_freeze = dataset_root / str(
            dataset.get("dataset_freeze", "dataset_freeze.json")
        )
        if not dataset_freeze.exists():
            raise FileNotFoundError(dataset_freeze)
        datasets[dataset_name] = {
            "dataset_root": _relative(root, dataset_root),
            "dataset_freeze": _relative(root, dataset_freeze),
            "dataset_freeze_sha256": file_sha256(dataset_freeze),
            "partitions": list(partitions),
        }
        normalized_dataset = {
            **dataset,
            "dataset_root": _relative(root, dataset_root),
        }
        for seed in seeds:
            payload = _manifest_payload(
                campaign=campaign,
                dataset_name=dataset_name,
                dataset=normalized_dataset,
                group="stochastic",
                seed=seed,
                methods=_stochastic_methods(
                    root,
                    campaign,
                    diffusion_checkpoints[seed],
                    direct_checkpoints[seed],
                ),
            )
            manifest_from_mapping(payload["experiment"])
            path = manifest_root / f"{payload['experiment']['name']}.yaml"
            path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            entries.append(
                {
                    "dataset": dataset_name,
                    "group": "stochastic",
                    "seed": seed,
                    "path": _relative(root, path),
                    "sha256": file_sha256(path),
                }
            )
        optimization_seed = int(campaign["optimization_seed"])
        payload = _manifest_payload(
            campaign=campaign,
            dataset_name=dataset_name,
            dataset=normalized_dataset,
            group="optimization",
            seed=optimization_seed,
            methods=_optimization_methods(campaign),
        )
        manifest_from_mapping(payload["experiment"])
        path = manifest_root / f"{payload['experiment']['name']}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        entries.append(
            {
                "dataset": dataset_name,
                "group": "optimization",
                "seed": optimization_seed,
                "path": _relative(root, path),
                "sha256": file_sha256(path),
            }
        )

    lock = {
        "schema_version": "1.0",
        "scope": SEALED_CAMPAIGN_SCOPE,
        "name": str(campaign["name"]),
        "config_path": _relative(root, Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        "seeds": list(seeds),
        "optimization_seed": int(campaign["optimization_seed"]),
        "diffusion_checkpoint_freeze": _relative(root, diffusion_freeze_path),
        "diffusion_checkpoint_freeze_sha256": file_sha256(diffusion_freeze_path),
        "direct_checkpoint_freeze": _relative(root, direct_freeze_path),
        "direct_checkpoint_freeze_sha256": file_sha256(direct_freeze_path),
        "datasets": datasets,
        "contract": {
            key: campaign[key]
            for key in (
                "device",
                "deterministic",
                "milp_time_limit_seconds",
                "proposal_count",
                "proposal_batch_size",
                "diffusion_reverse_steps",
                "repair_max_moves",
                "fallback_max_search_nodes",
            )
        },
        "manifests": entries,
    }
    lock_path = _resolve(root, campaign["lock_path"])
    write_json(lock_path, lock)
    return verify_sealed_campaign_lock(lock_path, implementation_root=root)


def verify_sealed_campaign_lock(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Verify every dataset, checkpoint-freeze, and manifest hash in the lock."""

    root = Path(implementation_root).resolve()
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("scope") != SEALED_CAMPAIGN_SCOPE:
        raise ValueError("Unsupported sealed campaign lock scope.")
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("diffusion_checkpoint_freeze", "diffusion_checkpoint_freeze_sha256"),
        ("direct_checkpoint_freeze", "direct_checkpoint_freeze_sha256"),
    ):
        path = _resolve(root, lock[path_key])
        if file_sha256(path) != lock[hash_key]:
            raise ValueError(f"Sealed campaign hash mismatch: {path}")
    verify_checkpoint_freeze(_resolve(root, lock["diffusion_checkpoint_freeze"]))
    verify_checkpoint_freeze(_resolve(root, lock["direct_checkpoint_freeze"]))
    for dataset in lock["datasets"].values():
        path = _resolve(root, dataset["dataset_freeze"])
        if file_sha256(path) != dataset["dataset_freeze_sha256"]:
            raise ValueError(f"Dataset freeze hash mismatch: {path}")
        audit_dataset_freeze(_resolve(root, dataset["dataset_root"]))
    for entry in lock["manifests"]:
        path = _resolve(root, entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Experiment manifest hash mismatch: {path}")
        manifest = load_experiment_manifest(path)
        if {"train", "validation"} & set(manifest.partitions):
            raise ValueError("Locked final manifest contains a non-test partition.")
    return lock


def run_sealed_campaign(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
    groups: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run selected locked groups and atomically record resumable run paths."""

    root = Path(implementation_root).resolve()
    lock = verify_sealed_campaign_lock(lock_path, implementation_root=root)
    requested = None if groups is None else set(groups)
    valid_groups = {
        f"{entry['dataset']}-{entry['group']}" for entry in lock["manifests"]
    }
    if requested is not None and not requested <= valid_groups:
        raise ValueError(f"Unknown sealed campaign groups: {sorted(requested - valid_groups)}")
    output_root = _resolve(
        root,
        load_config(_resolve(root, lock["config_path"]))["campaign"]["output_root"],
    )
    index_path = output_root / "run_index.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.exists()
        else {"schema_version": "1.0", "scope": SEALED_CAMPAIGN_SCOPE, "runs": {}}
    )
    for entry in lock["manifests"]:
        group = f"{entry['dataset']}-{entry['group']}"
        if requested is not None and group not in requested:
            continue
        manifest_path = _resolve(root, entry["path"])
        key = entry["path"]
        existing = index["runs"].get(key)
        if existing is not None:
            run = Path(existing["run_directory"])
            if (
                run.exists()
                and (run / "summary.json").exists()
                and file_sha256(run / "summary.json") == existing["summary_sha256"]
                and file_sha256(run / "resolved_manifest.json")
                == existing["resolved_manifest_sha256"]
                and file_sha256(run / "records.jsonl") == existing["records_sha256"]
            ):
                continue
        manifest = load_experiment_manifest(manifest_path)
        run = evaluate_experiment(manifest, implementation_root=root)
        index["runs"][key] = {
            **entry,
            "run_directory": str(run.resolve()),
            "summary_sha256": file_sha256(run / "summary.json"),
            "resolved_manifest_sha256": file_sha256(run / "resolved_manifest.json"),
            "records_sha256": file_sha256(run / "records.jsonl"),
        }
        write_json(index_path, index)
    return index


def finalize_sealed_campaign(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Verify all runs, aggregate five-seed groups, and freeze final evidence."""

    root = Path(implementation_root).resolve()
    lock = verify_sealed_campaign_lock(lock_path, implementation_root=root)
    campaign = load_config(_resolve(root, lock["config_path"]))["campaign"]
    output_root = _resolve(root, campaign["output_root"])
    index_path = output_root / "run_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if set(index["runs"]) != {entry["path"] for entry in lock["manifests"]}:
        raise ValueError("Sealed campaign is incomplete.")
    for record in index["runs"].values():
        run = Path(record["run_directory"])
        for filename, hash_key in (
            ("summary.json", "summary_sha256"),
            ("resolved_manifest.json", "resolved_manifest_sha256"),
            ("records.jsonl", "records_sha256"),
        ):
            if file_sha256(run / filename) != record[hash_key]:
                raise ValueError(f"Completed run hash mismatch: {run / filename}")

    aggregates: dict[str, str] = {}
    for dataset_name in lock["datasets"]:
        runs = [
            Path(record["run_directory"])
            for record in index["runs"].values()
            if record["dataset"] == dataset_name and record["group"] == "stochastic"
        ]
        destination = output_root / f"{dataset_name}_stochastic_five_seed.json"
        aggregate_run_directories(runs, output=destination)
        aggregates[dataset_name] = str(destination.resolve())
    evidence = {
        "schema_version": "1.0",
        "scope": "phase6d_c_final_evidence",
        "campaign_lock": str(Path(lock_path).resolve()),
        "campaign_lock_sha256": file_sha256(lock_path),
        "run_index": str(index_path.resolve()),
        "run_index_sha256": file_sha256(index_path),
        "aggregates": {
            name: {"path": path, "sha256": file_sha256(path)}
            for name, path in aggregates.items()
        },
        "runs": list(index["runs"].values()),
    }
    write_json(output_root / "final_evidence_freeze.json", evidence)
    return evidence
