"""Locked, resumable Phase 6E-B retraining and final-ID ablation campaign."""

from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.training import audit_dataset_freeze

from .aggregation import aggregate_run_directories
from .evaluation import evaluate_experiment
from .schema import file_sha256, load_experiment_manifest, manifest_from_mapping
from .training_aggregation import (
    aggregate_and_freeze_training_runs,
    audit_training_run,
    verify_checkpoint_freeze,
)

PHASE6EB_SCOPE = "phase6e_b_locked_training_ablation"
PHASE6EB_TRAINING_EVIDENCE_SCOPE = "phase6e_b_training_evidence"
PHASE6EB_EVALUATION_SCOPE = "phase6e_b_locked_final_id_evaluation"
PHASE6EB_FINAL_EVIDENCE_SCOPE = "phase6e_b_final_evidence"
TARGET_MODES = {"energy", "uniform", "best"}
PRIORITIES = {"anchor", "primary", "secondary", "diagnostic"}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Phase 6E-B path must stay inside implementation root: {path}") from error


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checkpoint_map(
    freeze: Mapping[str, Any],
    *,
    expected_seeds: Iterable[int] | None = None,
) -> dict[int, Path]:
    selected = None if expected_seeds is None else {int(seed) for seed in expected_seeds}
    mapping: dict[int, Path] = {}
    for record in freeze["runs"]:
        seed = int(record["seed"])
        if selected is not None and seed not in selected:
            continue
        path = Path(record["best_checkpoint"]).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if str(payload.get("model_kind", "diffusion")) != "diffusion":
            raise ValueError(f"Phase 6E-B requires a diffusion checkpoint: {path}")
        mapping[seed] = path
    if selected is not None and set(mapping) != selected:
        raise ValueError("Checkpoint freeze does not cover the requested Phase 6E-B seeds.")
    return mapping


def _validate_variant(variant_id: str, payload: Mapping[str, Any]) -> None:
    if not variant_id.strip():
        raise ValueError("Phase 6E-B variant IDs must be nonempty.")
    if str(payload["priority"]) not in PRIORITIES:
        raise ValueError(f"Unsupported Phase 6E-B priority for {variant_id!r}.")
    if str(payload["target_sampling"]) not in TARGET_MODES:
        raise ValueError(f"Unsupported target sampling for {variant_id!r}.")
    for key in ("capacity_weight", "link_weight"):
        if float(payload[key]) < 0:
            raise ValueError(f"{variant_id!r} has a negative {key}.")
    if bool(payload.get("anchor", False)) != (str(payload["priority"]) == "anchor"):
        raise ValueError(f"{variant_id!r} has an inconsistent anchor declaration.")


def _training_config(
    base: Mapping[str, Any],
    *,
    variant_id: str,
    variant: Mapping[str, Any],
    seed: int,
    run_name: str,
    steps: int,
    device: str,
    dataset_root: str,
    smoke_instance_limit: int | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    training = config["training"]
    training.update(
        {
            "run_name": run_name,
            "dataset_root": dataset_root,
            "seed": int(seed),
            "device": device,
            "steps": int(steps),
            "target_sampling": str(variant["target_sampling"]),
            "capacity_weight": float(variant["capacity_weight"]),
            "link_weight": float(variant["link_weight"]),
            "phase6eb_variant": variant_id,
        }
    )
    if set(training["train_partitions"]) != {"train"}:
        raise ValueError("Phase 6E-B training must use only the train partition.")
    if set(training["validation_partitions"]) != {"validation"}:
        raise ValueError("Phase 6E-B checkpoint selection must use only validation.")
    if smoke_instance_limit is not None:
        training.update(
            {
                "log_interval": 1,
                "denoising_validation_interval": int(steps),
                "constrained_validation_interval": int(steps),
                "checkpoint_interval": int(steps),
            }
        )
        config["constrained_validation"]["instance_limit"] = int(smoke_instance_limit)
    return config


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def prepare_phase6eb_campaign(
    config_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Generate deterministic smoke/full configs and lock all training inputs."""

    root = Path(implementation_root).resolve()
    config_path = Path(config_path).resolve()
    campaign = load_config(config_path)["campaign"]
    if str(campaign["schema_version"]) != "1.0":
        raise ValueError("Unsupported Phase 6E-B campaign schema.")
    seeds = tuple(int(seed) for seed in campaign["seeds"])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Phase 6E-B requires exactly three distinct seeds.")
    smoke_seed = int(campaign["smoke_seed"])
    if smoke_seed not in seeds:
        raise ValueError("Phase 6E-B smoke_seed must be one of the campaign seeds.")

    variants = campaign["variants"]
    if not isinstance(variants, Mapping) or len(variants) < 2:
        raise ValueError("Phase 6E-B requires multiple declared variants.")
    for variant_id, variant in variants.items():
        _validate_variant(str(variant_id), variant)
    anchors = [
        str(variant_id)
        for variant_id, variant in variants.items()
        if bool(variant.get("anchor", False))
    ]
    if anchors != ["energy_full"]:
        raise ValueError("Phase 6E-B requires energy_full as its single anchor.")

    base_path = _resolve(root, campaign["base_training_config"])
    base = load_config(base_path)
    dataset_root = _resolve(root, campaign["dataset_root"])
    dataset_freeze = dataset_root / str(campaign["dataset_freeze"])
    audit_dataset_freeze(dataset_root)
    anchor_path = _resolve(root, campaign["anchor_checkpoint_freeze"])
    anchor_freeze = verify_checkpoint_freeze(anchor_path)
    anchor_checkpoints = _checkpoint_map(anchor_freeze, expected_seeds=seeds)

    generated_root = _resolve(root, campaign["generated_config_root"])
    generated_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for variant_id, variant in variants.items():
        variant_id = str(variant_id)
        if bool(variant.get("anchor", False)):
            continue
        for seed in seeds:
            config = _training_config(
                base,
                variant_id=variant_id,
                variant=variant,
                seed=seed,
                run_name=f"phase6e-b-{variant_id}-seed{seed}",
                steps=int(campaign["expected_steps"]),
                device=str(campaign["device"]),
                dataset_root=_relative(root, dataset_root),
            )
            path = generated_root / "full" / f"{variant_id}-seed{seed}.yaml"
            _write_yaml(path, config)
            entries.append(
                {
                    "mode": "full",
                    "variant": variant_id,
                    "priority": str(variant["priority"]),
                    "seed": seed,
                    "expected_steps": int(campaign["expected_steps"]),
                    "path": _relative(root, path),
                    "sha256": file_sha256(path),
                }
            )
        smoke = _training_config(
            base,
            variant_id=variant_id,
            variant=variant,
            seed=smoke_seed,
            run_name=f"phase6e-b-smoke-{variant_id}-seed{smoke_seed}",
            steps=int(campaign["smoke_steps"]),
            device=str(campaign["device"]),
            dataset_root=_relative(root, dataset_root),
            smoke_instance_limit=int(campaign["smoke_validation_instances"]),
        )
        path = generated_root / "smoke" / f"{variant_id}-seed{smoke_seed}.yaml"
        _write_yaml(path, smoke)
        entries.append(
            {
                "mode": "smoke",
                "variant": variant_id,
                "priority": str(variant["priority"]),
                "seed": smoke_seed,
                "expected_steps": int(campaign["smoke_steps"]),
                "path": _relative(root, path),
                "sha256": file_sha256(path),
            }
        )

    lock = {
        "schema_version": "1.0",
        "scope": PHASE6EB_SCOPE,
        "name": str(campaign["name"]),
        "config_path": _relative(root, config_path),
        "config_sha256": file_sha256(config_path),
        "base_training_config": _relative(root, base_path),
        "base_training_config_sha256": file_sha256(base_path),
        "dataset_root": _relative(root, dataset_root),
        "dataset_freeze": _relative(root, dataset_freeze),
        "dataset_freeze_sha256": file_sha256(dataset_freeze),
        "anchor_checkpoint_freeze": _relative(root, anchor_path),
        "anchor_checkpoint_freeze_sha256": file_sha256(anchor_path),
        "seeds": list(seeds),
        "smoke_seed": smoke_seed,
        "variants": {
            str(variant_id): dict(variant) for variant_id, variant in variants.items()
        },
        "anchors": {
            str(seed): {
                "checkpoint": _relative(root, path),
                "sha256": file_sha256(path),
            }
            for seed, path in sorted(anchor_checkpoints.items())
        },
        "contract": {
            "expected_steps": int(campaign["expected_steps"]),
            "smoke_steps": int(campaign["smoke_steps"]),
            "device": str(campaign["device"]),
            "training_partitions": ["train"],
            "validation_partitions": ["validation"],
            "final_partitions": list(campaign["final_evaluation"]["partitions"]),
            "final_inference": dict(campaign["final_evaluation"]["inference"]),
        },
        "training_configs": entries,
    }
    lock_path = _resolve(root, campaign["lock_path"])
    write_json(lock_path, lock)
    return verify_phase6eb_lock(lock_path, implementation_root=root)


def verify_phase6eb_lock(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Verify every immutable input and generated training config."""

    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PHASE6EB_SCOPE:
        raise ValueError("Unsupported Phase 6E-B lock scope.")
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("base_training_config", "base_training_config_sha256"),
        ("dataset_freeze", "dataset_freeze_sha256"),
        ("anchor_checkpoint_freeze", "anchor_checkpoint_freeze_sha256"),
    ):
        path = _resolve(root, lock[path_key])
        if file_sha256(path) != lock[hash_key]:
            raise ValueError(f"Phase 6E-B lock hash mismatch: {path}")
    audit_dataset_freeze(_resolve(root, lock["dataset_root"]))
    anchor_freeze = verify_checkpoint_freeze(
        _resolve(root, lock["anchor_checkpoint_freeze"])
    )
    checkpoints = _checkpoint_map(anchor_freeze, expected_seeds=lock["seeds"])
    for seed, record in lock["anchors"].items():
        path = _resolve(root, record["checkpoint"])
        if checkpoints[int(seed)] != path or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Phase 6E-B anchor mismatch for seed {seed}.")

    seen: set[tuple[str, str, int]] = set()
    for entry in lock["training_configs"]:
        key = (str(entry["mode"]), str(entry["variant"]), int(entry["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate Phase 6E-B training entry: {key}")
        seen.add(key)
        path = _resolve(root, entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Phase 6E-B generated config hash mismatch: {path}")
        config = load_config(path)
        training = config["training"]
        variant = lock["variants"][entry["variant"]]
        if set(training["train_partitions"]) != {"train"} or set(
            training["validation_partitions"]
        ) != {"validation"}:
            raise ValueError("Phase 6E-B training config contains a test partition.")
        expected = {
            "seed": int(entry["seed"]),
            "steps": int(entry["expected_steps"]),
            "target_sampling": str(variant["target_sampling"]),
            "capacity_weight": float(variant["capacity_weight"]),
            "link_weight": float(variant["link_weight"]),
            "phase6eb_variant": str(entry["variant"]),
        }
        for key_name, expected_value in expected.items():
            if training[key_name] != expected_value:
                raise ValueError(f"Phase 6E-B config mismatch for {key_name}: {path}")
    final_partitions = set(lock["contract"]["final_partitions"])
    if final_partitions != {"test_id"}:
        raise ValueError("Phase 6E-B final evaluation must be test_id only.")
    return lock


def _matching_run_directories(
    output_root: Path,
    *,
    run_name: str,
    variant: str,
    seed: int,
    expected_steps: int,
) -> list[Path]:
    matches: list[Path] = []
    if not output_root.exists():
        return matches
    for directory in output_root.iterdir():
        config_path = directory / "config.json"
        if not directory.is_dir() or not config_path.exists():
            continue
        try:
            config = _read_json(config_path)
            training = config["training"]
        except (KeyError, json.JSONDecodeError):
            continue
        if (
            str(training.get("run_name")) == run_name
            and str(training.get("phase6eb_variant")) == variant
            and int(training.get("seed", -1)) == seed
            and int(training.get("steps", -1)) == expected_steps
        ):
            matches.append(directory.resolve())
    return sorted(matches, key=lambda path: path.stat().st_mtime)


def _finite_metrics(run: Path) -> None:
    metrics_path = run / "metrics.jsonl"
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Training run has no metrics: {run}")
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise ValueError(f"Non-finite training metric {key!r} in {run}")
    if not any(record.get("split") == "validation_constrained" for record in records):
        raise ValueError(f"Training run lacks constrained validation: {run}")


def _audit_campaign_run(
    run: Path,
    *,
    expected_steps: int,
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    audited = audit_training_run(run, expected_steps=expected_steps)
    config = _read_json(run / "config.json")
    training = config["training"]
    for key, expected in (
        ("target_sampling", str(variant["target_sampling"])),
        ("capacity_weight", float(variant["capacity_weight"])),
        ("link_weight", float(variant["link_weight"])),
    ):
        if training[key] != expected:
            raise ValueError(f"Completed run has an unexpected {key}: {run}")
    _finite_metrics(run)
    return audited


def _record_completed_run(
    entry: Mapping[str, Any],
    run: Path,
    audited: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(entry),
        "run_directory": str(run.resolve()),
        "summary_sha256": file_sha256(run / "summary.json"),
        "metrics_sha256": file_sha256(run / "metrics.jsonl"),
        "best_checkpoint": str((run / "best_checkpoint.pt").resolve()),
        "best_checkpoint_sha256": file_sha256(run / "best_checkpoint.pt"),
        "best_step": int(audited["best_step"]),
        "best_metrics": dict(audited["best_metrics"]),
    }


def run_phase6eb_training(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
    mode: str,
    variants: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Run or resume selected smoke/full training entries."""

    if mode not in {"smoke", "full"}:
        raise ValueError("Phase 6E-B training mode must be 'smoke' or 'full'.")
    root = Path(implementation_root).resolve()
    lock = verify_phase6eb_lock(lock_path, implementation_root=root)
    campaign = load_config(_resolve(root, lock["config_path"]))["campaign"]
    requested_variants = None if variants is None else {str(value) for value in variants}
    requested_seeds = None if seeds is None else {int(value) for value in seeds}
    valid_variants = set(lock["variants"]) - {"energy_full"}
    if requested_variants is not None and not requested_variants <= valid_variants:
        raise ValueError(
            f"Unknown Phase 6E-B variants: {sorted(requested_variants - valid_variants)}"
        )
    if requested_seeds is not None and not requested_seeds <= set(lock["seeds"]):
        raise ValueError("Requested Phase 6E-B seeds are outside the locked contract.")
    if str(lock["contract"]["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Phase 6E-B requires CUDA, but torch.cuda.is_available() is false.")

    output_root = _resolve(
        root,
        campaign["smoke_output_root"] if mode == "smoke" else campaign["full_output_root"],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = _resolve(
        root,
        campaign["smoke_index_path"] if mode == "smoke" else campaign["full_index_path"],
    )
    index = (
        _read_json(index_path)
        if index_path.exists()
        else {
            "schema_version": "1.0",
            "scope": PHASE6EB_SCOPE,
            "mode": mode,
            "runs": {},
        }
    )
    entries = [
        entry
        for entry in lock["training_configs"]
        if entry["mode"] == mode
        and (requested_variants is None or entry["variant"] in requested_variants)
        and (requested_seeds is None or int(entry["seed"]) in requested_seeds)
    ]
    priority_order = {"primary": 0, "secondary": 1, "diagnostic": 2}
    entries.sort(
        key=lambda entry: (
            priority_order[str(entry["priority"])],
            str(entry["variant"]),
            int(entry["seed"]),
        )
    )
    trainer = root / "scripts" / "17_train_production.py"
    for entry in entries:
        key = entry["path"]
        existing = index["runs"].get(key)
        if existing is not None:
            run = Path(existing["run_directory"])
            if (
                run.exists()
                and file_sha256(run / "summary.json") == existing["summary_sha256"]
                and file_sha256(run / "metrics.jsonl") == existing["metrics_sha256"]
                and file_sha256(run / "best_checkpoint.pt")
                == existing["best_checkpoint_sha256"]
            ):
                continue

        config_path = _resolve(root, entry["path"])
        config = load_config(config_path)
        run_name = str(config["training"]["run_name"])
        matches = _matching_run_directories(
            output_root,
            run_name=run_name,
            variant=str(entry["variant"]),
            seed=int(entry["seed"]),
            expected_steps=int(entry["expected_steps"]),
        )
        completed: Path | None = None
        resume: Path | None = None
        for candidate in reversed(matches):
            summary_path = candidate / "summary.json"
            if not summary_path.exists():
                continue
            summary = _read_json(summary_path)
            if (
                summary.get("termination_reason") == "completed"
                and int(summary.get("completed_steps", -1)) == int(entry["expected_steps"])
            ):
                completed = candidate
                break
            if (
                summary.get("termination_reason") == "interrupted"
                and (candidate / "latest_checkpoint.pt").exists()
            ):
                resume = candidate / "latest_checkpoint.pt"
                break
            if summary.get("termination_reason") == "failed":
                raise RuntimeError(f"Refusing to resume a failed training run: {candidate}")

        if completed is None:
            command = [
                sys.executable,
                str(trainer),
                "--config",
                str(config_path),
                "--output",
                str(output_root),
            ]
            if resume is not None:
                command.extend(["--resume", str(resume)])
            subprocess.run(command, cwd=root, check=True)
            matches = _matching_run_directories(
                output_root,
                run_name=run_name,
                variant=str(entry["variant"]),
                seed=int(entry["seed"]),
                expected_steps=int(entry["expected_steps"]),
            )
            completed = next(
                (
                    candidate
                    for candidate in reversed(matches)
                    if (candidate / "summary.json").exists()
                    and _read_json(candidate / "summary.json").get("termination_reason")
                    == "completed"
                    and int(
                        _read_json(candidate / "summary.json").get(
                            "completed_steps", -1
                        )
                    )
                    == int(entry["expected_steps"])
                ),
                None,
            )
        if completed is None:
            raise RuntimeError(f"Phase 6E-B training did not complete: {entry['path']}")
        audited = _audit_campaign_run(
            completed,
            expected_steps=int(entry["expected_steps"]),
            variant=lock["variants"][entry["variant"]],
        )
        index["runs"][key] = _record_completed_run(entry, completed, audited)
        write_json(index_path, index)
    return index


def verify_phase6eb_training_evidence(
    evidence_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Verify the complete per-variant training evidence freeze."""

    root = Path(implementation_root).resolve()
    evidence = _read_json(evidence_path)
    if evidence.get("scope") != PHASE6EB_TRAINING_EVIDENCE_SCOPE:
        raise ValueError("Unsupported Phase 6E-B training evidence scope.")
    for path_key, hash_key in (
        ("campaign_lock", "campaign_lock_sha256"),
        ("full_run_index", "full_run_index_sha256"),
    ):
        path = _resolve(root, evidence[path_key])
        if file_sha256(path) != evidence[hash_key]:
            raise ValueError(f"Phase 6E-B training evidence hash mismatch: {path}")
    for record in evidence["variant_freezes"].values():
        path = _resolve(root, record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Phase 6E-B variant-freeze hash mismatch: {path}")
        verify_checkpoint_freeze(path)
    return evidence


def finalize_phase6eb_training(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Audit all full runs and freeze three checkpoints per variant."""

    root = Path(implementation_root).resolve()
    lock = verify_phase6eb_lock(lock_path, implementation_root=root)
    campaign = load_config(_resolve(root, lock["config_path"]))["campaign"]
    index_path = _resolve(root, campaign["full_index_path"])
    index = _read_json(index_path)
    expected_entries = {
        entry["path"]
        for entry in lock["training_configs"]
        if entry["mode"] == "full"
    }
    if set(index["runs"]) != expected_entries:
        missing = expected_entries - set(index["runs"])
        raise ValueError(f"Phase 6E-B full training is incomplete: {sorted(missing)}")

    anchor_freeze = verify_checkpoint_freeze(
        _resolve(root, lock["anchor_checkpoint_freeze"])
    )
    anchor_runs = {
        int(run["seed"]): Path(run["run_directory"])
        for run in anchor_freeze["runs"]
        if int(run["seed"]) in set(lock["seeds"])
    }
    output_root = _resolve(root, campaign["variant_freeze_root"])
    variant_freezes: dict[str, dict[str, str]] = {}
    for variant_id in lock["variants"]:
        if variant_id == "energy_full":
            runs = [anchor_runs[int(seed)] for seed in lock["seeds"]]
        else:
            records = [
                record
                for record in index["runs"].values()
                if record["variant"] == variant_id
            ]
            runs = [
                Path(record["run_directory"])
                for record in sorted(records, key=lambda value: int(value["seed"]))
            ]
        destination = output_root / variant_id
        aggregate_and_freeze_training_runs(
            runs,
            expected_seeds=lock["seeds"],
            expected_steps=int(lock["contract"]["expected_steps"]),
            output_directory=destination,
            scope="independent_training_seeds",
        )
        freeze_path = destination / "checkpoint_freeze.json"
        variant_freezes[variant_id] = {
            "path": _relative(root, freeze_path),
            "sha256": file_sha256(freeze_path),
        }
    evidence = {
        "schema_version": "1.0",
        "scope": PHASE6EB_TRAINING_EVIDENCE_SCOPE,
        "campaign_lock": _relative(root, Path(lock_path).resolve()),
        "campaign_lock_sha256": file_sha256(lock_path),
        "full_run_index": _relative(root, index_path),
        "full_run_index_sha256": file_sha256(index_path),
        "seeds": list(lock["seeds"]),
        "variant_freezes": variant_freezes,
    }
    destination = _resolve(root, campaign["training_evidence_path"])
    write_json(destination, evidence)
    return verify_phase6eb_training_evidence(destination, implementation_root=root)


def _evaluation_methods(
    root: Path,
    campaign: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[int, Path]],
    seed: int,
) -> list[dict[str, Any]]:
    inference = dict(campaign["final_evaluation"]["inference"])
    proposal_group = str(campaign["final_evaluation"]["proposal_group"])
    return [
        {
            "method_id": variant_id,
            "kind": "learned_hybrid",
            "checkpoint": _relative(root, by_seed[seed]),
            "proposal_group": proposal_group,
            "inference": inference,
        }
        for variant_id, by_seed in checkpoints.items()
    ]


def prepare_phase6eb_evaluation(
    training_evidence_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Generate final-ID manifests only after all training freezes verify."""

    root = Path(implementation_root).resolve()
    evidence = verify_phase6eb_training_evidence(
        training_evidence_path, implementation_root=root
    )
    training_lock = verify_phase6eb_lock(
        _resolve(root, _read_json(training_evidence_path)["campaign_lock"]),
        implementation_root=root,
    )
    campaign = load_config(_resolve(root, training_lock["config_path"]))["campaign"]
    checkpoints: dict[str, dict[int, Path]] = {}
    for variant_id, record in evidence["variant_freezes"].items():
        freeze = verify_checkpoint_freeze(_resolve(root, record["path"]))
        checkpoints[variant_id] = _checkpoint_map(
            freeze, expected_seeds=training_lock["seeds"]
        )

    manifest_root = _resolve(root, campaign["generated_evaluation_root"])
    manifest_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed in training_lock["seeds"]:
        methods = _evaluation_methods(root, campaign, checkpoints, int(seed))
        method_ids = [method["method_id"] for method in methods]
        payload = {
            "experiment": {
                "schema_version": "1.0",
                "name": f"phase6e-b-final-id-seed{seed}",
                "dataset_root": str(campaign["dataset_root"]),
                "dataset_freeze": str(campaign["dataset_freeze"]),
                "partitions": list(campaign["final_evaluation"]["partitions"]),
                "seed": int(seed),
                "device": str(campaign["device"]),
                "deterministic": bool(campaign["final_evaluation"]["deterministic"]),
                "output_root": str(campaign["evaluation_output_root"]),
                "methods": methods,
                "claims": [
                    {
                        "claim_id": "energy_weighting",
                        "question": "Does energy-weighted target sampling improve verified deployment quality?",
                        "hypothesis": "Energy weighting lowers exact latency gap relative to uniform target sampling.",
                        "comparison": ["energy_full", "uniform_full"],
                        "primary_metric": "gap_to_pool_best",
                    },
                    {
                        "claim_id": "soft_guidance",
                        "question": "Does soft constraint guidance improve raw proposal quality and reduce fallback burden?",
                        "hypothesis": "Full guidance improves raw feasibility and post-processing burden relative to no guidance.",
                        "comparison": ["energy_full", "energy_no_guidance"],
                        "primary_metric": "raw_feasible_rate",
                    },
                    {
                        "claim_id": "all_phase6eb_variants",
                        "question": "How do all locked training variants compare under a common final-ID evaluator?",
                        "hypothesis": "The variants expose mechanism-specific quality and feasibility tradeoffs.",
                        "comparison": method_ids,
                        "primary_metric": "gap_to_pool_best",
                    },
                ],
            }
        }
        manifest_from_mapping(payload["experiment"])
        path = manifest_root / f"phase6e-b-final-id-seed{seed}.yaml"
        _write_yaml(path, payload)
        entries.append(
            {
                "seed": int(seed),
                "path": _relative(root, path),
                "sha256": file_sha256(path),
            }
        )
    lock = {
        "schema_version": "1.0",
        "scope": PHASE6EB_EVALUATION_SCOPE,
        "training_evidence": _relative(root, Path(training_evidence_path).resolve()),
        "training_evidence_sha256": file_sha256(training_evidence_path),
        "seeds": list(training_lock["seeds"]),
        "variants": list(training_lock["variants"]),
        "manifests": entries,
    }
    lock_path = _resolve(root, campaign["evaluation_lock_path"])
    write_json(lock_path, lock)
    return verify_phase6eb_evaluation_lock(lock_path, implementation_root=root)


def verify_phase6eb_evaluation_lock(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PHASE6EB_EVALUATION_SCOPE:
        raise ValueError("Unsupported Phase 6E-B evaluation lock scope.")
    evidence_path = _resolve(root, lock["training_evidence"])
    if file_sha256(evidence_path) != lock["training_evidence_sha256"]:
        raise ValueError("Phase 6E-B training evidence changed after evaluation lock.")
    verify_phase6eb_training_evidence(evidence_path, implementation_root=root)
    for entry in lock["manifests"]:
        path = _resolve(root, entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Phase 6E-B evaluation manifest hash mismatch: {path}")
        manifest = load_experiment_manifest(path)
        if set(manifest.partitions) != {"test_id"}:
            raise ValueError("Phase 6E-B evaluation lock contains a non-ID partition.")
        if {method.method_id for method in manifest.methods} != set(lock["variants"]):
            raise ValueError("Phase 6E-B evaluation manifest has incomplete variants.")
    return lock


def run_phase6eb_evaluation(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Run or reuse each locked three-seed final-ID evaluation."""

    root = Path(implementation_root).resolve()
    lock = verify_phase6eb_evaluation_lock(lock_path, implementation_root=root)
    evidence = verify_phase6eb_training_evidence(
        _resolve(root, lock["training_evidence"]), implementation_root=root
    )
    training_lock = verify_phase6eb_lock(
        _resolve(root, evidence["campaign_lock"]), implementation_root=root
    )
    campaign = load_config(_resolve(root, training_lock["config_path"]))["campaign"]
    index_path = _resolve(root, campaign["evaluation_index_path"])
    index = (
        _read_json(index_path)
        if index_path.exists()
        else {
            "schema_version": "1.0",
            "scope": PHASE6EB_EVALUATION_SCOPE,
            "runs": {},
        }
    )
    for entry in lock["manifests"]:
        key = entry["path"]
        existing = index["runs"].get(key)
        if existing is not None:
            run = Path(existing["run_directory"])
            if all(
                file_sha256(run / filename) == existing[hash_key]
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


def _format(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _metric(block: Mapping[str, Any], name: str, statistic: str = "mean") -> float | None:
    value = block.get(name)
    return None if value is None else float(value[statistic])


def _write_results(
    aggregate: Mapping[str, Any],
    *,
    output_root: Path,
    run_directories: Iterable[Path],
) -> dict[str, Any]:
    source_counts: dict[str, dict[str, int]] = {}
    for run in run_directories:
        for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            variant = str(record["method_id"])
            source = str(record["source"])
            source_counts.setdefault(variant, {})
            source_counts[variant][source] = (
                source_counts[variant].get(source, 0) + 1
            )
    rows: list[dict[str, Any]] = []
    for variant, metrics in aggregate["methods"].items():
        counts = source_counts[variant]
        total_sources = sum(counts.values())
        rows.append(
            {
                "variant": variant,
                "gap_percent_mean": 100.0 * float(_metric(metrics, "gap_to_pool_best") or 0.0),
                "gap_percent_std": 100.0 * float(
                    _metric(metrics, "gap_to_pool_best", "std") or 0.0
                ),
                "raw_feasible_percent_mean": 100.0
                * float(_metric(metrics, "raw_feasible_rate") or 0.0),
                "raw_feasible_percent_std": 100.0
                * float(_metric(metrics, "raw_feasible_rate", "std") or 0.0),
                "raw_selected_percent": 100.0
                * counts.get("raw", 0)
                / total_sources,
                "repair_selected_percent": 100.0
                * counts.get("repair", 0)
                / total_sources,
                "fallback_selected_percent": 100.0
                * counts.get("fallback", 0)
                / total_sources,
                "repair_success_rate_mean": _metric(metrics, "repair_success_rate"),
                "total_seconds_mean": _metric(metrics, "total_seconds"),
                "success_percent_mean": 100.0
                * float(_metric(metrics, "success") or 0.0),
            }
        )
    csv_path = output_root / "phase6eb_ablation_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_variant = {row["variant"]: row for row in rows}
    energy = by_variant["energy_full"]
    uniform = by_variant["uniform_full"]
    no_guidance = by_variant["energy_no_guidance"]
    no_capacity = by_variant["energy_no_capacity"]
    no_link = by_variant["energy_no_link"]
    best = by_variant["best_full"]
    lines = [
        "# Phase 6E-B Frozen Retraining-Ablation Results",
        "",
        "All values come from the locked three-seed final-ID campaign. Final output",
        "quality uses exact weighted end-to-end latency; raw feasibility and",
        "fallback burden diagnose the proposal stage before the hard pipeline.",
        "",
        "| Variant | Gap (%) | Raw feasible (%) | Selected raw/repair/fallback (%) | Time (s) |",
        "|---|---:|---:|---:|---:|",
        *[
            "| {variant} | {gap:.3f} +/- {gap_std:.3f} | "
            "{raw:.2f} +/- {raw_std:.2f} | {raw_selected:.2f}/{repair:.2f}/{fallback:.2f} | {time} |".format(
                variant=row["variant"],
                gap=row["gap_percent_mean"],
                gap_std=row["gap_percent_std"],
                raw=row["raw_feasible_percent_mean"],
                raw_std=row["raw_feasible_percent_std"],
                raw_selected=row["raw_selected_percent"],
                repair=row["repair_selected_percent"],
                fallback=row["fallback_selected_percent"],
                time=_format(row["total_seconds_mean"], 4),
            )
            for row in rows
        ],
        "",
        "## Locked Comparisons",
        "",
        "- Energy versus uniform target sampling changes mean exact gap by "
        f"{uniform['gap_percent_mean'] - energy['gap_percent_mean']:+.3f} percentage points "
        "(positive favors energy weighting).",
        "- Full versus no soft guidance changes mean raw feasibility by "
        f"{energy['raw_feasible_percent_mean'] - no_guidance['raw_feasible_percent_mean']:+.2f} "
        "percentage points (positive favors full guidance).",
        "- Full versus no soft guidance changes the fraction of final outputs",
        "  selected from fallback by "
        f"{energy['fallback_selected_percent'] - no_guidance['fallback_selected_percent']:+.2f} "
        "percentage points (negative would favor full guidance).",
        "- Best-only supervision attains a mean gap "
        f"{energy['gap_percent_mean'] - best['gap_percent_mean']:+.3f} percentage points "
        "below energy-weighted supervision in this campaign; energy weighting is",
        "  therefore not uniformly superior to a single-best target.",
        "- Removing link guidance changes mean gap by "
        f"{no_link['gap_percent_mean'] - energy['gap_percent_mean']:+.3f} percentage points, "
        "whereas removing capacity guidance changes it by "
        f"{no_capacity['gap_percent_mean'] - energy['gap_percent_mean']:+.3f} percentage points.",
        "- These three-seed statistics support only the observed synthetic-ID",
        "  comparisons; they do not establish universal component necessity.",
        "",
    ]
    markdown_path = output_root / "PHASE6E_B_RESULTS.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "table": {"path": str(csv_path.resolve()), "sha256": file_sha256(csv_path)},
        "markdown": {
            "path": str(markdown_path.resolve()),
            "sha256": file_sha256(markdown_path),
        },
        "rows": rows,
    }


def finalize_phase6eb_evaluation(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Aggregate, interpret conservatively, and freeze final Phase 6E-B evidence."""

    root = Path(implementation_root).resolve()
    lock = verify_phase6eb_evaluation_lock(lock_path, implementation_root=root)
    evidence = verify_phase6eb_training_evidence(
        _resolve(root, lock["training_evidence"]), implementation_root=root
    )
    training_lock = verify_phase6eb_lock(
        _resolve(root, evidence["campaign_lock"]), implementation_root=root
    )
    campaign = load_config(_resolve(root, training_lock["config_path"]))["campaign"]
    index_path = _resolve(root, campaign["evaluation_index_path"])
    index = _read_json(index_path)
    if set(index["runs"]) != {entry["path"] for entry in lock["manifests"]}:
        raise ValueError("Phase 6E-B final-ID evaluation is incomplete.")
    runs = []
    for record in index["runs"].values():
        run = Path(record["run_directory"])
        for filename, hash_key in (
            ("summary.json", "summary_sha256"),
            ("resolved_manifest.json", "resolved_manifest_sha256"),
            ("records.jsonl", "records_sha256"),
        ):
            if file_sha256(run / filename) != record[hash_key]:
                raise ValueError(f"Phase 6E-B completed-run hash mismatch: {run}")
        runs.append(run)
    output_root = _resolve(root, campaign["evaluation_output_root"])
    aggregate_path = output_root / "phase6eb_three_seed_aggregate.json"
    aggregate = aggregate_run_directories(runs, output=aggregate_path)
    results = _write_results(
        aggregate,
        output_root=output_root,
        run_directories=runs,
    )
    final = {
        "schema_version": "1.0",
        "scope": PHASE6EB_FINAL_EVIDENCE_SCOPE,
        "evaluation_lock": _relative(root, Path(lock_path).resolve()),
        "evaluation_lock_sha256": file_sha256(lock_path),
        "training_evidence": lock["training_evidence"],
        "training_evidence_sha256": lock["training_evidence_sha256"],
        "evaluation_index": _relative(root, index_path),
        "evaluation_index_sha256": file_sha256(index_path),
        "aggregate": {
            "path": _relative(root, aggregate_path),
            "sha256": file_sha256(aggregate_path),
        },
        "table": {
            "path": _relative(root, Path(results["table"]["path"])),
            "sha256": results["table"]["sha256"],
        },
        "markdown": {
            "path": _relative(root, Path(results["markdown"]["path"])),
            "sha256": results["markdown"]["sha256"],
        },
        "runs": list(index["runs"].values()),
    }
    destination = _resolve(root, campaign["final_evidence_path"])
    write_json(destination, final)
    return final


def verify_phase6eb_final_evidence(
    evidence_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Verify every artifact in the final Phase 6E-B evidence package."""

    root = Path(implementation_root).resolve()
    evidence = _read_json(evidence_path)
    if evidence.get("scope") != PHASE6EB_FINAL_EVIDENCE_SCOPE:
        raise ValueError("Unsupported Phase 6E-B final evidence scope.")
    for path_key, hash_key in (
        ("evaluation_lock", "evaluation_lock_sha256"),
        ("training_evidence", "training_evidence_sha256"),
        ("evaluation_index", "evaluation_index_sha256"),
    ):
        path = _resolve(root, evidence[path_key])
        if file_sha256(path) != evidence[hash_key]:
            raise ValueError(f"Phase 6E-B final evidence hash mismatch: {path}")
    for artifact in ("aggregate", "table", "markdown"):
        record = evidence[artifact]
        path = _resolve(root, record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Phase 6E-B result artifact hash mismatch: {path}")
    for record in evidence["runs"]:
        run = Path(record["run_directory"])
        for filename, hash_key in (
            ("summary.json", "summary_sha256"),
            ("resolved_manifest.json", "resolved_manifest_sha256"),
            ("records.jsonl", "records_sha256"),
        ):
            if file_sha256(run / filename) != record[hash_key]:
                raise ValueError(f"Phase 6E-B run hash mismatch: {run / filename}")
    verify_phase6eb_evaluation_lock(
        _resolve(root, evidence["evaluation_lock"]),
        implementation_root=root,
    )
    return evidence
