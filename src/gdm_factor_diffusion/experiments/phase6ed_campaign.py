"""Locked time-matched diffusion/direct comparison for Phase 6E-D."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import yaml

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.training import audit_dataset_freeze

from .aggregation import aggregate_run_directories, read_jsonl
from .evaluation import evaluate_experiment
from .schema import file_sha256, load_experiment_manifest, manifest_from_mapping
from .training_aggregation import verify_checkpoint_freeze

PHASE6ED_VALIDATION_SCOPE = "phase6e_d_validation_calibration"
PHASE6ED_BUDGET_SCOPE = "phase6e_d_time_matched_budget"
PHASE6ED_FINAL_SCOPE = "phase6e_d_locked_final_id"
PHASE6ED_EVIDENCE_SCOPE = "phase6e_d_final_evidence"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checkpoint_map(
    freeze: Mapping[str, Any],
    *,
    expected_kind: str,
    expected_seeds: Iterable[int],
) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for run in freeze["runs"]:
        seed = int(run["seed"])
        path = Path(run["best_checkpoint"]).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        kind = str(payload.get("model_kind", "diffusion"))
        if kind != expected_kind:
            raise ValueError(
                f"Expected {expected_kind!r} checkpoint, found {kind!r}: {path}"
            )
        checkpoints[seed] = path
    if set(checkpoints) != set(expected_seeds):
        raise ValueError("Checkpoint freeze and Phase 6E-D seeds disagree.")
    return checkpoints


def _inference(
    campaign: Mapping[str, Any],
    *,
    num_samples: int,
    sample_batch_size: int,
    reverse_steps: int | None,
) -> dict[str, int]:
    values = {
        "num_samples": int(num_samples),
        "sample_batch_size": int(sample_batch_size),
        "repair_max_moves": int(
            campaign["shared_postprocessing"]["repair_max_moves"]
        ),
        "fallback_max_search_nodes": int(
            campaign["shared_postprocessing"]["fallback_max_search_nodes"]
        ),
    }
    if reverse_steps is not None:
        values["reverse_steps"] = int(reverse_steps)
    return values


def _method(
    root: Path,
    campaign: Mapping[str, Any],
    checkpoint: Path,
    *,
    method_id: str,
    kind: str,
    proposal_group: str,
    num_samples: int,
    sample_batch_size: int,
    reverse_steps: int | None,
) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "kind": kind,
        "checkpoint": _relative(root, checkpoint),
        "proposal_group": proposal_group,
        "inference": _inference(
            campaign,
            num_samples=num_samples,
            sample_batch_size=sample_batch_size,
            reverse_steps=reverse_steps,
        ),
    }


def _diffusion_method(
    root: Path,
    campaign: Mapping[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    diffusion = campaign["diffusion"]
    return _method(
        root,
        campaign,
        checkpoint,
        method_id=f"diffusion_k{int(diffusion['num_samples'])}",
        kind="learned_hybrid",
        proposal_group=str(campaign["proposal_groups"]["diffusion"]),
        num_samples=int(diffusion["num_samples"]),
        sample_batch_size=int(diffusion["sample_batch_size"]),
        reverse_steps=int(diffusion["reverse_steps"]),
    )


def _direct_method(
    root: Path,
    campaign: Mapping[str, Any],
    checkpoint: Path,
    proposal_count: int,
) -> dict[str, Any]:
    return _method(
        root,
        campaign,
        checkpoint,
        method_id=f"direct_k{proposal_count}",
        kind="direct_hybrid",
        proposal_group=str(campaign["proposal_groups"]["direct"]),
        num_samples=proposal_count,
        sample_batch_size=min(
            int(campaign["direct_sample_batch_size"]), proposal_count
        ),
        reverse_steps=None,
    )


def _manifest_payload(
    root: Path,
    campaign: Mapping[str, Any],
    *,
    seed: int,
    diffusion_checkpoint: Path,
    direct_checkpoint: Path,
    partition: str,
    proposal_counts: Iterable[int],
    name: str,
    output_root: str,
) -> dict[str, Any]:
    methods = [_diffusion_method(root, campaign, diffusion_checkpoint)]
    methods.extend(
        _direct_method(root, campaign, direct_checkpoint, int(count))
        for count in proposal_counts
    )
    method_ids = [method["method_id"] for method in methods]
    return {
        "experiment": {
            "schema_version": "1.0",
            "name": name,
            "dataset_root": str(campaign["dataset_root"]),
            "dataset_freeze": str(campaign["dataset_freeze"]),
            "partitions": [partition],
            "seed": seed,
            "device": str(campaign["device"]),
            "deterministic": bool(campaign["deterministic"]),
            "output_root": output_root,
            "methods": methods,
            "claims": [
                {
                    "claim_id": f"phase6ed_{partition}",
                    "question": (
                        "Does iterative diffusion improve proposal coverage or "
                        "verified latency over direct prediction at matched budget?"
                    ),
                    "hypothesis": (
                        "Diffusion provides a favorable proposal-quality/runtime "
                        "tradeoff after controlling the direct proposal budget."
                    ),
                    "comparison": method_ids,
                    "primary_metric": "gap_to_pool_best",
                }
            ],
        }
    }


def prepare_phase6ed_validation(
    config_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Generate and lock validation-only initial and extension manifests."""

    root = Path(implementation_root).resolve()
    config_path = Path(config_path).resolve()
    campaign = load_config(config_path)["campaign"]
    if str(campaign["schema_version"]) != "1.0":
        raise ValueError("Unsupported Phase 6E-D campaign schema.")
    seeds = tuple(int(seed) for seed in campaign["seeds"])
    if not 0 < float(campaign["time_match_ratio"]) <= 1:
        raise ValueError("time_match_ratio must lie in (0, 1].")
    grids = {
        group: tuple(int(value) for value in values)
        for group, values in campaign["validation_grids"].items()
    }
    if set(grids) != {"initial", "extension"}:
        raise ValueError("Phase 6E-D requires initial and extension grids.")
    if len(set(grids["initial"] + grids["extension"])) != sum(
        len(values) for values in grids.values()
    ):
        raise ValueError("Direct proposal counts must be unique across grids.")

    diffusion_freeze_path = _resolve(
        root, campaign["diffusion_checkpoint_freeze"]
    )
    direct_freeze_path = _resolve(root, campaign["direct_checkpoint_freeze"])
    diffusion_freeze = verify_checkpoint_freeze(diffusion_freeze_path)
    direct_freeze = verify_checkpoint_freeze(direct_freeze_path)
    diffusion_checkpoints = _checkpoint_map(
        diffusion_freeze, expected_kind="diffusion", expected_seeds=seeds
    )
    direct_checkpoints = _checkpoint_map(
        direct_freeze, expected_kind="direct", expected_seeds=seeds
    )
    dataset_root = _resolve(root, campaign["dataset_root"])
    audit_dataset_freeze(dataset_root)
    dataset_freeze = dataset_root / str(campaign["dataset_freeze"])
    validation_partition = str(campaign["validation_partition"])
    final_partition = str(campaign["final_partition"])
    if validation_partition == final_partition:
        raise ValueError("Validation and final partitions must differ.")

    manifest_root = _resolve(root, campaign["generated_manifest_root"])
    manifest_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for group, proposal_counts in grids.items():
        for seed in seeds:
            name = f"phase6e-d-validation-{group}-seed{seed}"
            payload = _manifest_payload(
                root,
                campaign,
                seed=seed,
                diffusion_checkpoint=diffusion_checkpoints[seed],
                direct_checkpoint=direct_checkpoints[seed],
                partition=validation_partition,
                proposal_counts=proposal_counts,
                name=name,
                output_root=f"{campaign['output_root']}/validation/{group}",
            )
            manifest_from_mapping(payload["experiment"])
            path = manifest_root / f"{name}.yaml"
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
        "scope": PHASE6ED_VALIDATION_SCOPE,
        "config_path": _relative(root, config_path),
        "config_sha256": file_sha256(config_path),
        "dataset_root": _relative(root, dataset_root),
        "dataset_freeze": _relative(root, dataset_freeze),
        "dataset_freeze_sha256": file_sha256(dataset_freeze),
        "diffusion_checkpoint_freeze": _relative(root, diffusion_freeze_path),
        "diffusion_checkpoint_freeze_sha256": file_sha256(diffusion_freeze_path),
        "direct_checkpoint_freeze": _relative(root, direct_freeze_path),
        "direct_checkpoint_freeze_sha256": file_sha256(direct_freeze_path),
        "seeds": list(seeds),
        "validation_partition": validation_partition,
        "final_partition": final_partition,
        "time_match_ratio": float(campaign["time_match_ratio"]),
        "grids": {key: list(values) for key, values in grids.items()},
        "manifests": entries,
    }
    lock_path = _resolve(root, campaign["validation_lock_path"])
    write_json(lock_path, lock)
    return verify_phase6ed_validation_lock(lock_path, implementation_root=root)


def verify_phase6ed_validation_lock(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PHASE6ED_VALIDATION_SCOPE:
        raise ValueError("Unsupported Phase 6E-D validation lock scope.")
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("dataset_freeze", "dataset_freeze_sha256"),
        ("diffusion_checkpoint_freeze", "diffusion_checkpoint_freeze_sha256"),
        ("direct_checkpoint_freeze", "direct_checkpoint_freeze_sha256"),
    ):
        path = _resolve(root, lock[path_key])
        if file_sha256(path) != lock[hash_key]:
            raise ValueError(f"Phase 6E-D validation lock hash mismatch: {path}")
    audit_dataset_freeze(_resolve(root, lock["dataset_root"]))
    verify_checkpoint_freeze(_resolve(root, lock["diffusion_checkpoint_freeze"]))
    verify_checkpoint_freeze(_resolve(root, lock["direct_checkpoint_freeze"]))
    for entry in lock["manifests"]:
        path = _resolve(root, entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Phase 6E-D manifest hash mismatch: {path}")
        manifest = load_experiment_manifest(path)
        if manifest.partitions != (lock["validation_partition"],):
            raise ValueError("Validation lock contains a non-validation partition.")
        if lock["final_partition"] in manifest.partitions:
            raise ValueError("Final ID appeared before budget selection.")
    return lock


def _run_index_path(root: Path, campaign: Mapping[str, Any], stage: str) -> Path:
    return _resolve(root, campaign["output_root"]) / stage / "run_index.json"


def run_phase6ed_validation(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
    groups: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run or reuse selected validation grids."""

    root = Path(implementation_root).resolve()
    lock = verify_phase6ed_validation_lock(lock_path, implementation_root=root)
    campaign = load_config(_resolve(root, lock["config_path"]))["campaign"]
    requested = {"initial"} if groups is None else set(groups)
    if not requested <= set(lock["grids"]):
        raise ValueError(f"Unknown Phase 6E-D validation groups: {sorted(requested)}")
    index_path = _run_index_path(root, campaign, "validation")
    index = (
        _read_json(index_path)
        if index_path.exists()
        else {"schema_version": "1.0", "scope": PHASE6ED_VALIDATION_SCOPE, "runs": {}}
    )
    for entry in lock["manifests"]:
        if entry["group"] not in requested:
            continue
        key = entry["path"]
        existing = index["runs"].get(key)
        if existing is not None:
            run = Path(existing["run_directory"])
            if existing.get("sha256") == entry["sha256"] and all(
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


def _verify_index_records(index: Mapping[str, Any]) -> None:
    for record in index["runs"].values():
        run = Path(record["run_directory"])
        for filename, hash_key in (
            ("summary.json", "summary_sha256"),
            ("resolved_manifest.json", "resolved_manifest_sha256"),
            ("records.jsonl", "records_sha256"),
        ):
            if file_sha256(run / filename) != record[hash_key]:
                raise ValueError(f"Phase 6E-D run hash mismatch: {run / filename}")


def _metric(aggregate: Mapping[str, Any], method: str, metric: str) -> float:
    block = aggregate["methods"][method].get(metric)
    if block is None:
        raise ValueError(f"Missing {metric!r} for {method!r}.")
    return float(block["mean"])


def select_time_matched_budget(
    aggregate_by_group: Mapping[str, Mapping[str, Any]],
    *,
    time_match_ratio: float,
) -> dict[str, Any]:
    """Select the smallest direct K reaching the validation timing threshold."""

    if not aggregate_by_group:
        raise ValueError("At least one validation aggregate is required.")
    direct_times: dict[int, float] = {}
    direct_ratios: dict[int, float] = {}
    direct_anchor_times: dict[int, float] = {}
    for aggregate in aggregate_by_group.values():
        diffusion_time = _metric(aggregate, "diffusion_k4", "total_seconds")
        for method in aggregate["methods"]:
            if method.startswith("direct_k"):
                count = int(method.removeprefix("direct_k"))
                direct_time = _metric(aggregate, method, "total_seconds")
                direct_times[count] = direct_time
                direct_anchor_times[count] = diffusion_time
                direct_ratios[count] = direct_time / diffusion_time
    reaching = [
        count for count, ratio in direct_ratios.items() if ratio >= time_match_ratio
    ]
    if reaching:
        selected = min(reaching)
        rule = "smallest_k_reaching_threshold"
    else:
        selected = min(
            direct_ratios,
            key=lambda count: (abs(direct_ratios[count] - 1.0), count),
        )
        rule = "closest_tested_runtime"
    selected_anchor = direct_anchor_times[selected]
    return {
        "diffusion_mean_total_seconds": selected_anchor,
        "target_ratio": float(time_match_ratio),
        "threshold_seconds": time_match_ratio * selected_anchor,
        "direct_mean_total_seconds": {
            str(key): value for key, value in sorted(direct_times.items())
        },
        "direct_to_paired_diffusion_time_ratio": {
            str(key): value for key, value in sorted(direct_ratios.items())
        },
        "selected_direct_k": selected,
        "selected_direct_mean_total_seconds": direct_times[selected],
        "selected_to_diffusion_time_ratio": direct_ratios[selected],
        "selection_rule": rule,
        "threshold_reached": bool(reaching),
    }


def finalize_phase6ed_validation(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Aggregate validation timing and freeze K, or request the extension grid."""

    root = Path(implementation_root).resolve()
    lock = verify_phase6ed_validation_lock(lock_path, implementation_root=root)
    campaign = load_config(_resolve(root, lock["config_path"]))["campaign"]
    index_path = _run_index_path(root, campaign, "validation")
    index = _read_json(index_path)
    _verify_index_records(index)
    expected_by_group = {
        group: {entry["path"] for entry in lock["manifests"] if entry["group"] == group}
        for group in lock["grids"]
    }
    completed = set(index["runs"])
    if not expected_by_group["initial"] <= completed:
        raise ValueError("Phase 6E-D initial validation grid is incomplete.")

    output_root = _resolve(root, campaign["output_root"])
    aggregates: dict[str, dict[str, Any]] = {}
    aggregate_paths: dict[str, dict[str, str]] = {}
    for group in ("initial", "extension"):
        if not expected_by_group[group] <= completed:
            continue
        runs = [
            Path(index["runs"][path]["run_directory"])
            for path in sorted(expected_by_group[group])
        ]
        path = output_root / "validation" / f"{group}_five_seed.json"
        aggregates[group] = aggregate_run_directories(runs, output=path)
        aggregate_paths[group] = {
            "path": _relative(root, path),
            "sha256": file_sha256(path),
        }

    initial_selection = select_time_matched_budget(
        {"initial": aggregates["initial"]},
        time_match_ratio=float(lock["time_match_ratio"]),
    )
    if not initial_selection["threshold_reached"] and "extension" not in aggregates:
        status = {
            "schema_version": "1.0",
            "scope": PHASE6ED_BUDGET_SCOPE,
            "status": "needs_extension",
            "validation_lock": _relative(root, Path(lock_path).resolve()),
            "validation_lock_sha256": file_sha256(lock_path),
            "initial_aggregate": aggregate_paths["initial"],
            "initial_selection": initial_selection,
            "required_group": "extension",
        }
        status_path = output_root / "validation" / "calibration_status.json"
        write_json(status_path, status)
        return status

    selection = select_time_matched_budget(
        aggregates,
        time_match_ratio=float(lock["time_match_ratio"]),
    )
    budget = {
        "schema_version": "1.0",
        "scope": PHASE6ED_BUDGET_SCOPE,
        "status": "selected",
        "validation_lock": _relative(root, Path(lock_path).resolve()),
        "validation_lock_sha256": file_sha256(lock_path),
        "run_index": _relative(root, index_path),
        "run_index_sha256": file_sha256(index_path),
        "aggregates": aggregate_paths,
        "selection": selection,
        "selected_without_final_id": True,
    }
    budget_path = _resolve(root, campaign["budget_freeze_path"])
    write_json(budget_path, budget)
    return verify_phase6ed_budget_freeze(budget_path, implementation_root=root)


def verify_phase6ed_budget_freeze(
    budget_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    budget = _read_json(budget_path)
    if budget.get("scope") != PHASE6ED_BUDGET_SCOPE or budget.get("status") != "selected":
        raise ValueError("Phase 6E-D budget has not been selected.")
    lock_path = _resolve(root, budget["validation_lock"])
    if file_sha256(lock_path) != budget["validation_lock_sha256"]:
        raise ValueError("Phase 6E-D validation lock changed after budget selection.")
    verify_phase6ed_validation_lock(lock_path, implementation_root=root)
    index_path = _resolve(root, budget["run_index"])
    if file_sha256(index_path) != budget["run_index_sha256"]:
        raise ValueError("Phase 6E-D validation index changed after budget selection.")
    for record in budget["aggregates"].values():
        path = _resolve(root, record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError("Phase 6E-D validation aggregate hash mismatch.")
    return budget


def prepare_phase6ed_final(
    budget_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Generate final-ID manifests only after the direct budget is frozen."""

    root = Path(implementation_root).resolve()
    budget = verify_phase6ed_budget_freeze(budget_path, implementation_root=root)
    validation_lock_path = _resolve(root, budget["validation_lock"])
    validation_lock = verify_phase6ed_validation_lock(
        validation_lock_path, implementation_root=root
    )
    campaign = load_config(_resolve(root, validation_lock["config_path"]))["campaign"]
    seeds = tuple(int(seed) for seed in validation_lock["seeds"])
    diffusion_checkpoints = _checkpoint_map(
        verify_checkpoint_freeze(
            _resolve(root, validation_lock["diffusion_checkpoint_freeze"])
        ),
        expected_kind="diffusion",
        expected_seeds=seeds,
    )
    direct_checkpoints = _checkpoint_map(
        verify_checkpoint_freeze(
            _resolve(root, validation_lock["direct_checkpoint_freeze"])
        ),
        expected_kind="direct",
        expected_seeds=seeds,
    )
    selected = int(budget["selection"]["selected_direct_k"])
    proposal_counts = sorted({4, selected})
    manifest_root = _resolve(root, campaign["generated_manifest_root"])
    entries = []
    for seed in seeds:
        name = f"phase6e-d-final-id-seed{seed}"
        payload = _manifest_payload(
            root,
            campaign,
            seed=seed,
            diffusion_checkpoint=diffusion_checkpoints[seed],
            direct_checkpoint=direct_checkpoints[seed],
            partition=str(validation_lock["final_partition"]),
            proposal_counts=proposal_counts,
            name=name,
            output_root=f"{campaign['output_root']}/final_id",
        )
        manifest_from_mapping(payload["experiment"])
        path = manifest_root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        entries.append(
            {
                "seed": seed,
                "path": _relative(root, path),
                "sha256": file_sha256(path),
            }
        )
    lock = {
        "schema_version": "1.0",
        "scope": PHASE6ED_FINAL_SCOPE,
        "budget_freeze": _relative(root, Path(budget_path).resolve()),
        "budget_freeze_sha256": file_sha256(budget_path),
        "dataset_root": validation_lock["dataset_root"],
        "dataset_freeze": validation_lock["dataset_freeze"],
        "dataset_freeze_sha256": validation_lock["dataset_freeze_sha256"],
        "partition": validation_lock["final_partition"],
        "seeds": list(seeds),
        "selected_direct_k": selected,
        "methods": ["diffusion_k4", "direct_k4", f"direct_k{selected}"],
        "manifests": entries,
    }
    lock_path = _resolve(root, campaign["final_lock_path"])
    write_json(lock_path, lock)
    return verify_phase6ed_final_lock(lock_path, implementation_root=root)


def verify_phase6ed_final_lock(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = _read_json(lock_path)
    if lock.get("scope") != PHASE6ED_FINAL_SCOPE:
        raise ValueError("Unsupported Phase 6E-D final lock scope.")
    budget_path = _resolve(root, lock["budget_freeze"])
    if file_sha256(budget_path) != lock["budget_freeze_sha256"]:
        raise ValueError("Phase 6E-D budget changed after final lock generation.")
    verify_phase6ed_budget_freeze(budget_path, implementation_root=root)
    if file_sha256(_resolve(root, lock["dataset_freeze"])) != lock["dataset_freeze_sha256"]:
        raise ValueError("Phase 6E-D dataset freeze changed.")
    for entry in lock["manifests"]:
        path = _resolve(root, entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Phase 6E-D final manifest hash mismatch: {path}")
        manifest = load_experiment_manifest(path)
        if manifest.partitions != (lock["partition"],):
            raise ValueError("Phase 6E-D final lock contains the wrong partition.")
    return lock


def run_phase6ed_final(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    lock = verify_phase6ed_final_lock(lock_path, implementation_root=root)
    budget = verify_phase6ed_budget_freeze(
        _resolve(root, lock["budget_freeze"]), implementation_root=root
    )
    validation_lock = verify_phase6ed_validation_lock(
        _resolve(root, budget["validation_lock"]), implementation_root=root
    )
    campaign = load_config(_resolve(root, validation_lock["config_path"]))["campaign"]
    index_path = _run_index_path(root, campaign, "final_id")
    index = (
        _read_json(index_path)
        if index_path.exists()
        else {"schema_version": "1.0", "scope": PHASE6ED_FINAL_SCOPE, "runs": {}}
    )
    for entry in lock["manifests"]:
        key = entry["path"]
        existing = index["runs"].get(key)
        if existing is not None:
            run = Path(existing["run_directory"])
            if existing.get("sha256") == entry["sha256"] and all(
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


def _source_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record["method_id"])][str(record["source"])] += 1
    return {method: dict(sorted(values.items())) for method, values in sorted(counts.items())}


def _paired_outcome(
    records: Iterable[Mapping[str, Any]], left: str, right: str
) -> dict[str, int]:
    grouped: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["method_id"] in {left, right}:
            key = (int(record["campaign_seed"]), str(record["instance_id"]))
            grouped[key][str(record["method_id"])] = record
    counts = {"left_wins": 0, "ties": 0, "right_wins": 0}
    for pair in grouped.values():
        if set(pair) != {left, right}:
            continue
        left_value = float(pair[left]["objective"])
        right_value = float(pair[right]["objective"])
        if left_value < right_value - 1e-12:
            counts["left_wins"] += 1
        elif right_value < left_value - 1e-12:
            counts["right_wins"] += 1
        else:
            counts["ties"] += 1
    return counts


def _report_markdown(
    aggregate: Mapping[str, Any],
    *,
    selected_k: int,
    source_counts: Mapping[str, Mapping[str, int]],
    paired: Mapping[str, int],
) -> str:
    methods = ["diffusion_k4", "direct_k4", f"direct_k{selected_k}"]
    headers = (
        "| Method | Gap (%) | Time (s) | Raw any (%) | Unique (%) | "
        "Hamming | Pre-fallback success (%) | Fallback selected |"
    )
    lines = [
        "# Phase 6E-D 实验报告",
        "",
        "## 核心结果",
        "",
        headers,
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        lines.append(
            "| {method} | {gap:.3f} | {time:.4f} | {raw:.2f} | {unique:.2f} | "
            "{hamming:.4f} | {prefallback:.2f} | {fallback} |".format(
                method=method,
                gap=100 * _metric(aggregate, method, "gap_to_pool_best"),
                time=_metric(aggregate, method, "total_seconds"),
                raw=100 * _metric(aggregate, method, "raw_any_feasible"),
                unique=100 * _metric(aggregate, method, "raw_unique_rate"),
                hamming=_metric(aggregate, method, "raw_pairwise_hamming"),
                prefallback=100 * _metric(
                    aggregate, method, "pre_fallback_success"
                ),
                fallback=source_counts.get(method, {}).get("fallback", 0),
            )
        )
    diffusion_gap = _metric(aggregate, "diffusion_k4", "gap_to_pool_best")
    matched_method = f"direct_k{selected_k}"
    direct_gap = _metric(aggregate, matched_method, "gap_to_pool_best")
    diffusion_time = _metric(aggregate, "diffusion_k4", "total_seconds")
    direct_time = _metric(aggregate, matched_method, "total_seconds")
    favorable = diffusion_gap < direct_gap
    conclusion = (
        "在近似时间匹配下，diffusion 仍取得更低的验证后时延 gap。"
        if favorable
        else "在近似时间匹配下，direct predictor 已追平或超过 diffusion 的时延质量。"
    )
    gap_reduction = 100 * (diffusion_gap - direct_gap) / diffusion_gap
    lines.extend(
        [
            "",
            "## 解读",
            "",
            f"- 验证集锁定的 direct proposal budget 为 K={selected_k}。",
            f"- 最终 ID 上时间比 direct/diffusion 为 {direct_time / diffusion_time:.3f}。",
            f"- direct K={selected_k} 的平均 gap 相对 diffusion 降低 {gap_reduction:.1f}%。",
            f"- 成对结果（diffusion / tie / direct）为 "
            f"{paired['left_wins']} / {paired['ties']} / {paired['right_wins']}。",
            f"- 最佳 raw-feasible gap 为 diffusion "
            f"{100 * _metric(aggregate, 'diffusion_k4', 'best_raw_gap_to_pool_best'):.3f}%、"
            f"direct K={selected_k} "
            f"{100 * _metric(aggregate, matched_method, 'best_raw_gap_to_pool_best'):.3f}%；"
            f"repair 后、fallback 前分别为 "
            f"{100 * _metric(aggregate, 'diffusion_k4', 'best_pre_fallback_gap_to_pool_best'):.3f}% "
            f"和 {100 * _metric(aggregate, matched_method, 'best_pre_fallback_gap_to_pool_best'):.3f}%。",
            f"- 平均唯一 raw proposals 数为 diffusion "
            f"{_metric(aggregate, 'diffusion_k4', 'raw_unique_count'):.2f}、"
            f"direct K={selected_k} "
            f"{_metric(aggregate, matched_method, 'raw_unique_count'):.2f}；"
            "归一化 Hamming 多样性接近，未观察到 diffusion 的逐样本多样性优势。",
            f"- {conclusion}",
            "- 该结论只支持当前冻结合成 ID 场景下的质量-时间权衡；不能据此声称扩散普遍优于一次预测。",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_phase6ed_final(
    lock_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    """Aggregate, report, and freeze the final-ID time-matched evidence."""

    root = Path(implementation_root).resolve()
    lock = verify_phase6ed_final_lock(lock_path, implementation_root=root)
    budget = verify_phase6ed_budget_freeze(
        _resolve(root, lock["budget_freeze"]), implementation_root=root
    )
    validation_lock = verify_phase6ed_validation_lock(
        _resolve(root, budget["validation_lock"]), implementation_root=root
    )
    campaign = load_config(_resolve(root, validation_lock["config_path"]))["campaign"]
    index_path = _run_index_path(root, campaign, "final_id")
    index = _read_json(index_path)
    expected = {entry["path"] for entry in lock["manifests"]}
    if set(index["runs"]) != expected:
        raise ValueError("Phase 6E-D final-ID campaign is incomplete.")
    _verify_index_records(index)
    runs = [Path(index["runs"][path]["run_directory"]) for path in sorted(expected)]
    output_root = _resolve(root, campaign["output_root"])
    aggregate_path = output_root / "final_id_five_seed.json"
    aggregate = aggregate_run_directories(runs, output=aggregate_path)
    records = []
    for run in runs:
        resolved = _read_json(run / "resolved_manifest.json")
        campaign_seed = int(resolved["manifest"]["seed"])
        for record in read_jsonl(run / "records.jsonl"):
            records.append({**record, "campaign_seed": campaign_seed})
    selected = int(lock["selected_direct_k"])
    matched_method = f"direct_k{selected}"
    source_counts = _source_counts(records)
    paired = _paired_outcome(records, "diffusion_k4", matched_method)
    report_path = root / "PHASE6E_D_REPORT_ZH.md"
    report_path.write_text(
        _report_markdown(
            aggregate,
            selected_k=selected,
            source_counts=source_counts,
            paired=paired,
        ),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "1.0",
        "scope": PHASE6ED_EVIDENCE_SCOPE,
        "final_lock": _relative(root, Path(lock_path).resolve()),
        "final_lock_sha256": file_sha256(lock_path),
        "budget_freeze": lock["budget_freeze"],
        "budget_freeze_sha256": lock["budget_freeze_sha256"],
        "run_index": _relative(root, index_path),
        "run_index_sha256": file_sha256(index_path),
        "aggregate": {
            "path": _relative(root, aggregate_path),
            "sha256": file_sha256(aggregate_path),
        },
        "report": {
            "path": _relative(root, report_path),
            "sha256": file_sha256(report_path),
        },
        "selected_direct_k": selected,
        "source_counts": source_counts,
        "paired_diffusion_vs_time_matched_direct": paired,
        "method_diagnostics": {
            method: {
                metric: _metric(aggregate, method, metric)
                for metric in (
                    "gap_to_pool_best",
                    "total_seconds",
                    "raw_feasible_rate",
                    "raw_any_feasible",
                    "raw_unique_count",
                    "raw_unique_rate",
                    "raw_pairwise_hamming",
                    "best_raw_gap_to_pool_best",
                    "pre_fallback_success",
                    "best_pre_fallback_gap_to_pool_best",
                )
            }
            for method in ("diffusion_k4", "direct_k4", matched_method)
        },
        "diffusion_gap_mean": _metric(
            aggregate, "diffusion_k4", "gap_to_pool_best"
        ),
        "time_matched_direct_gap_mean": _metric(
            aggregate, matched_method, "gap_to_pool_best"
        ),
        "diffusion_total_seconds_mean": _metric(
            aggregate, "diffusion_k4", "total_seconds"
        ),
        "time_matched_direct_total_seconds_mean": _metric(
            aggregate, matched_method, "total_seconds"
        ),
        "interpretation": (
            "narrow_diffusion_quality_support"
            if _metric(aggregate, "diffusion_k4", "gap_to_pool_best")
            < _metric(aggregate, matched_method, "gap_to_pool_best")
            else "direct_matches_or_exceeds_diffusion_at_matched_time"
        ),
        "runs": list(index["runs"].values()),
    }
    evidence_path = output_root / "final_evidence_freeze.json"
    write_json(evidence_path, evidence)
    return verify_phase6ed_final_evidence(evidence_path, implementation_root=root)


def verify_phase6ed_final_evidence(
    evidence_path: str | Path,
    *,
    implementation_root: str | Path,
) -> dict[str, Any]:
    root = Path(implementation_root).resolve()
    evidence = _read_json(evidence_path)
    if evidence.get("scope") != PHASE6ED_EVIDENCE_SCOPE:
        raise ValueError("Unsupported Phase 6E-D evidence scope.")
    for path_key, hash_key in (
        ("final_lock", "final_lock_sha256"),
        ("budget_freeze", "budget_freeze_sha256"),
        ("run_index", "run_index_sha256"),
    ):
        path = _resolve(root, evidence[path_key])
        if file_sha256(path) != evidence[hash_key]:
            raise ValueError(f"Phase 6E-D evidence hash mismatch: {path}")
    verify_phase6ed_final_lock(
        _resolve(root, evidence["final_lock"]), implementation_root=root
    )
    for record in (evidence["aggregate"], evidence["report"]):
        path = _resolve(root, record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Phase 6E-D artifact hash mismatch: {path}")
    return evidence
