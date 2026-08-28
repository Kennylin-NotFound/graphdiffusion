"""Evaluate frozen learned generators with proposal-conditioned recovery.

The runner reuses the frozen Phase 6F proposal samplers and changes only the
post-generation solver contract:

1. hard-verify every generated proposal;
2. select the best verified raw proposal when one exists;
3. otherwise attempt bounded recovery from the failed proposals;
4. hard-verify recovered placements and evaluate exact latency;
5. report failure when both branches produce no verified placement.

Legacy repair and independent constructive fallback are disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.data.dataset import load_manifest, load_manifest_instance
from gdm_factor_diffusion.experiments.schema import file_sha256
from gdm_factor_diffusion.graph.batch_adapter import infer_feature_schema
from gdm_factor_diffusion.inference import InferenceConfig, sample_random_proposals


SCOPE = "phase7_proposal_recovery_evaluation"
RECORD_SCOPE = f"{SCOPE}_record"
EVIDENCE_SCOPE = f"{SCOPE}_evidence"

DEFAULT_SEEDS = tuple(range(2026070111, 2026070121))
DEFAULT_METHODS = (
    "random_k64",
    "direct_b64_t1",
    "sequential_b64_t1",
    "masked_k1_t1_mix",
    "masked_k8_t1",
)
METHOD_LABELS = {
    "random_k64": "Random + Recovery",
    "direct_b64_t1": "Direct GNN",
    "sequential_b64_t1": "Sequential GNN",
    "masked_k1_t1_mix": "Masked deterministic",
    "masked_k8_t1": "Masked Diffusion",
}

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "main_evaluation": {
        "dataset_root": "artifacts/datasets/phase6e-e-stage38-sealed",
        "partitions": ("sealed_test_id",),
    },
    "controlled_shift": {
        "dataset_root": "artifacts/datasets/phase6e-e-controlled-shift",
        "reference_records": "artifacts/phase6e-e-controlled-shift-evaluation-10seed/records/2026070111",
        "partitions": (
            "test_id_reference",
            "shift_high_sharing",
            "shift_low_compatibility",
            "shift_tight_capacity",
            "shift_unseen_workflow",
        ),
    },
    "cross_scale": {
        "dataset_root": "artifacts/datasets/phase6c-final-scale",
        "partitions": ("scale_medium", "scale_large", "scale_extra_large"),
    },
    "realistic_simulation": {
        "dataset_root": "artifacts/datasets/phase6e-e-realistic-profile",
        "reference_records": "artifacts/phase6e-e-realistic-profile-evaluation-10seed/records/2026070111",
        "partitions": ("profile_id", "profile_branched", "profile_high_sharing"),
    },
}


def _load_base_module() -> Any:
    path = Path(__file__).with_name("92_probe_neural_decoding_enhancements.py")
    spec = importlib.util.spec_from_file_location("phase7_sampling_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen sampling helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_module()
_ORIGINAL_SAMPLE_METHOD = _BASE._sample_method
_ORIGINAL_AUDIT_DATASET_FREEZE = _BASE.audit_dataset_freeze
_ORIGINAL_LABELED_DATASET = _BASE.LabeledDeploymentDataset


@dataclass(frozen=True, slots=True)
class _ReferencePool:
    """Minimal immutable reference required by the evaluation runner."""

    pool_best: float
    original_size: int

    @property
    def latencies(self) -> np.ndarray:
        return np.asarray([self.pool_best], dtype=np.float64)

    @property
    def size(self) -> int:
        return self.original_size


@dataclass(frozen=True, slots=True)
class _ReferenceItem:
    instance: Any
    pool: _ReferencePool
    partition: str


def _reference_record_digest(record_root: Path) -> str:
    """Hash record paths and contents so reused MILP references remain auditable."""

    digest = hashlib.sha256()
    paths = sorted(record_root.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No reference records found under {record_root}.")
    for path in paths:
        relative = path.relative_to(record_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def _reference_spec_for_root(dataset_root: Path) -> Mapping[str, Any] | None:
    for spec in DATASET_SPECS.values():
        if Path(spec["dataset_root"]).name == dataset_root.name:
            return spec if "reference_records" in spec else None
    return None


def _reference_record_root(dataset_root: Path, spec: Mapping[str, Any]) -> Path:
    artifacts_root = dataset_root.parents[1]
    relative = Path(str(spec["reference_records"]))
    parts = relative.parts
    if len(parts) < 2 or parts[0] != "artifacts":
        raise ValueError("reference_records must be relative to the implementation root.")
    return artifacts_root.joinpath(*parts[1:]).resolve()


def _prepare_reference_contract(dataset_root: Path) -> None:
    """Freeze instance files and the reused per-instance MILP reference records."""

    if (dataset_root / "solution_pool_manifest.json").exists():
        return
    spec = _reference_spec_for_root(dataset_root)
    if spec is None:
        raise FileNotFoundError(
            f"No solution pools or reference-record contract for {dataset_root}."
        )
    record_root = _reference_record_root(dataset_root, spec)
    source_hashes: set[str] = set()
    for path in sorted(record_root.rglob("*.json")):
        payload = _read_json(path)
        value = payload.get("dataset_freeze_sha256")
        if value:
            source_hashes.add(str(value))
    if len(source_hashes) != 1:
        raise ValueError(
            f"Reference records must share one source dataset hash: {record_root}."
        )
    freeze_path = dataset_root / "dataset_freeze.json"
    contract = {
        "schema_version": "1.0",
        "scope": "phase7_reference_record_dataset_freeze",
        "dataset_name": dataset_root.name,
        "core_sha256": {
            "catalog.json": file_sha256(dataset_root / "catalog.json"),
            "manifest.json": file_sha256(dataset_root / "manifest.json"),
        },
        "reference_records": str(record_root),
        "reference_records_sha256": _reference_record_digest(record_root),
        "source_dataset_freeze_sha256": next(iter(source_hashes)),
        "reference_training_seed": DEFAULT_SEEDS[0],
    }
    if freeze_path.exists() and _read_json(freeze_path) != contract:
        raise ValueError(f"Existing reference dataset freeze disagrees: {freeze_path}")
    if not freeze_path.exists():
        write_json(freeze_path, contract)


def _audit_dataset_freeze(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root)
    if not (root / "dataset_freeze.json").exists():
        _prepare_reference_contract(root)
    freeze = _ORIGINAL_AUDIT_DATASET_FREEZE(root)
    if freeze.get("scope") == "phase7_reference_record_dataset_freeze":
        record_root = Path(str(freeze["reference_records"]))
        actual = _reference_record_digest(record_root)
        if actual != freeze["reference_records_sha256"]:
            raise ValueError(f"Reference-record hash mismatch: {record_root}")
    return freeze


class _ReferenceRecordDataset:
    """Load checksum-protected instances with frozen MILP objective references."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        partitions: Sequence[str],
        verify_checksum: bool = True,
        require_freeze: bool = False,
    ) -> None:
        self.root = Path(dataset_root)
        self.verify_checksum = verify_checksum
        if require_freeze:
            _audit_dataset_freeze(self.root)
        spec = _reference_spec_for_root(self.root)
        if spec is None:
            raise ValueError(f"No reference-record specification for {self.root}.")
        record_root = _reference_record_root(self.root, spec)
        selected = set(partitions)
        references: dict[str, dict[str, Any]] = {}
        for partition in selected:
            for path in sorted((record_root / partition).glob("*.json")):
                payload = _read_json(path)
                instance_id = str(payload["instance_id"])
                if str(payload["partition"]) != partition:
                    raise ValueError(f"Reference partition mismatch: {path}")
                pool_best = float(payload["pool_best"])
                pool_size = int(payload["pool_size"])
                if not math.isfinite(pool_best) or pool_best < 0 or pool_size < 1:
                    raise ValueError(f"Invalid MILP reference: {path}")
                references[instance_id] = {
                    "partition": partition,
                    "pool_best": pool_best,
                    "pool_size": pool_size,
                }

        manifest = load_manifest(self.root / "manifest.json")
        unknown = selected - set(manifest["partitions"])
        if unknown:
            raise ValueError(f"Unknown dataset partitions: {sorted(unknown)}")
        self.entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for entry in manifest["instances"]:
            if entry["partition"] not in selected:
                continue
            reference = references.get(str(entry["instance_id"]))
            if reference is None:
                raise ValueError(
                    f"Missing frozen MILP reference for {entry['instance_id']!r}."
                )
            self.entries.append((entry, reference))
        if len(self.entries) != len(references):
            raise ValueError("Reference records and selected manifest entries disagree.")
        if not self.entries:
            raise ValueError("No reference-record instances were selected.")
        schema_instances = [
            load_manifest_instance(
                self.root,
                entry,
                verify_checksum=self.verify_checksum,
            )
            for entry, _ in self.entries
        ]
        self.feature_schema = infer_feature_schema(schema_instances)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> _ReferenceItem:
        entry, reference = self.entries[index]
        instance = load_manifest_instance(
            self.root,
            entry,
            verify_checksum=self.verify_checksum,
        )
        return _ReferenceItem(
            instance=instance,
            pool=_ReferencePool(
                pool_best=float(reference["pool_best"]),
                original_size=int(reference["pool_size"]),
            ),
            partition=str(entry["partition"]),
        )


def _dataset_factory(*args: Any, **kwargs: Any) -> Any:
    dataset_root = Path(args[0] if args else kwargs["dataset_root"])
    if (dataset_root / "solution_pool_manifest.json").exists():
        return _ORIGINAL_LABELED_DATASET(*args, **kwargs)
    _prepare_reference_contract(dataset_root)
    return _ReferenceRecordDataset(*args, **kwargs)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, value: str | Path) -> str:
    return Path(value).resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_csv(value: str | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return tuple(default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_seeds(value: str | None) -> tuple[int, ...]:
    seeds = tuple(int(part) for part in _parse_csv(value, [str(x) for x in DEFAULT_SEEDS]))
    unknown = sorted(set(seeds) - set(DEFAULT_SEEDS))
    if unknown:
        raise ValueError(f"Unsupported frozen seeds: {unknown}")
    return seeds


def _method_spec(method_id: str) -> dict[str, Any]:
    if method_id == "random_k64":
        return {
            "family": "random",
            "budget": 0,
            "samples": 64,
            "temperature": None,
            "policy": "compatible_prior",
            "stochastic": True,
        }
    return _BASE._method_spec(method_id)


def _protocol(
    *,
    root: Path,
    settings: tuple[str, ...],
    selected_seeds: tuple[int, ...],
    methods: tuple[str, ...],
    training_freeze: Path,
    sequential_checkpoint_root: Path,
    output_root: Path,
    device: str,
    max_instances_per_partition: int | None,
    fallback_max_search_nodes: int,
    skip_missing_datasets: bool,
) -> dict[str, Any]:
    del fallback_max_search_nodes
    return {
        "schema_version": "1.0",
        "scope": SCOPE,
        "settings": list(settings),
        "selected_seeds": list(selected_seeds),
        "methods": {method: _method_spec(method) for method in methods},
        "training_freeze": _relative(root, training_freeze),
        "sequential_checkpoint_root": _relative(root, sequential_checkpoint_root),
        "output_root": _relative(root, output_root),
        "device": device,
        "evaluation_seed": 2026071611,
        "sample_batch_size": 8,
        "max_instances_per_partition": max_instances_per_partition,
        "skip_missing_datasets": bool(skip_missing_datasets),
        "policy": {
            "name": "verified_raw_then_bounded_proposal_recovery",
            "enable_repair": False,
            "enable_recovery": True,
            "enable_fallback": False,
            "recovery_candidate_limit": 4,
            "recovery_max_released_services": 4,
        },
        "dataset_specs": {
            name: {
                "dataset_root": spec["dataset_root"],
                "partitions": list(spec["partitions"]),
                **(
                    {"reference_records": spec["reference_records"]}
                    if "reference_records" in spec
                    else {}
                ),
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
        enable_repair=False,
        enable_recovery=True,
        enable_fallback=False,
        always_include_fallback=False,
        recovery_candidate_limit=int(policy["recovery_candidate_limit"]),
        recovery_max_released_services=int(policy["recovery_max_released_services"]),
    )


def _sample_method(**kwargs: Any) -> dict[str, Any]:
    spec = kwargs["spec"]
    if spec["family"] != "random":
        return _ORIGINAL_SAMPLE_METHOD(**kwargs)

    instance = kwargs["instance"]
    generator = kwargs["generator"]
    samples = int(spec["samples"])
    cpu_generator = torch.Generator(device="cpu").manual_seed(generator.initial_seed())
    start = perf_counter()
    proposals = sample_random_proposals(
        instance,
        num_samples=samples,
        generator=cpu_generator,
    )
    return {
        "proposals": proposals,
        "probabilities": None,
        "seconds": perf_counter() - start,
        "model_forwards": 0,
        "completed_rate": 1.0,
        "realized_budget": 0,
    }


def _payload(
    result: Any,
    pool_best: float,
    sampled: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metrics = result.metrics
    raw_objective = metrics.get("best_raw_objective")
    total_proposals = int(metrics.get("num_raw_proposals", 0))
    raw_feasible = int(metrics.get("raw_feasible_count", 0))
    return {
        "success": bool(result.success),
        "source": result.source if result.source is not None else "failure",
        "objective": result.objective,
        "gap_to_pool_best": (
            None if result.objective is None else float(result.objective) / pool_best - 1.0
        ),
        "raw_any_feasible": bool(metrics.get("raw_any_feasible", False)),
        "best_raw_objective": raw_objective,
        "raw_gap_to_pool_best": (
            None if raw_objective is None else float(raw_objective) / pool_best - 1.0
        ),
        "num_raw_proposals": total_proposals,
        "raw_feasible_count": raw_feasible,
        "raw_feasible_rate": (
            raw_feasible / total_proposals if total_proposals else None
        ),
        "raw_unique_rate": metrics.get("raw_unique_rate"),
        "raw_pairwise_hamming": metrics.get("raw_pairwise_hamming"),
        "recovery_invoked": bool(metrics.get("recovery_invoked", False)),
        "recovery_attempts": int(metrics.get("recovery_attempts", 0)),
        "recovery_successes": int(metrics.get("recovery_successes", 0)),
        "recovery_released_services": int(
            metrics.get("recovery_released_services", 0)
        ),
        "recovery_completion_steps": int(
            metrics.get("recovery_completion_steps", 0)
        ),
        "recovery_failure_reasons": metrics.get("recovery_failure_reasons", {}),
        "sampling_seconds": float(metrics.get("sampling_seconds", 0.0)),
        "verification_seconds": float(metrics.get("verification_seconds", 0.0)),
        "recovery_seconds": float(metrics.get("recovery_seconds", 0.0)),
        "exact_evaluation_seconds": float(
            metrics.get("exact_evaluation_seconds", 0.0)
        ),
        "total_seconds": float(metrics.get("total_seconds", 0.0)),
        "model_forwards": None if sampled is None else sampled.get("model_forwards"),
        "completed_rate": None if sampled is None else sampled.get("completed_rate"),
        "realized_budget": None if sampled is None else sampled.get("realized_budget"),
    }


def _configure_base() -> None:
    _BASE.SCOPE = SCOPE
    _BASE.RECORD_SCOPE = RECORD_SCOPE
    _BASE.EVIDENCE_SCOPE = EVIDENCE_SCOPE
    _BASE.DATASET_SPECS = DATASET_SPECS
    _BASE._protocol = _protocol
    _BASE._inference_config = _inference_config
    _BASE._sample_method = _sample_method
    _BASE._payload = _payload
    _BASE.audit_dataset_freeze = _audit_dataset_freeze
    _BASE.LabeledDeploymentDataset = _dataset_factory


def _records(output_root: Path) -> list[Mapping[str, Any]]:
    return [
        _read_json(path)
        for path in sorted((output_root / "records").rglob("*.json"))
    ]


def _finite(values: Sequence[float | None]) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def _mean(values: Sequence[float | None]) -> float | None:
    data = _finite(values)
    return mean(data) if data else None


def _std(values: Sequence[float | None]) -> float | None:
    data = _finite(values)
    if not data:
        return None
    return pstdev(data) if len(data) > 1 else 0.0


def _aggregate(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = len(payloads)
    total_proposals = sum(int(row["num_raw_proposals"]) for row in payloads)
    total_raw_feasible = sum(int(row["raw_feasible_count"]) for row in payloads)
    invoked = [row for row in payloads if row["recovery_invoked"]]
    recovered_records = [
        row for row in invoked if row["success"] and row["source"] == "recovery"
    ]
    source_counts: dict[str, int] = defaultdict(int)
    failure_reasons: dict[str, int] = defaultdict(int)
    for row in payloads:
        source_counts[str(row["source"])] += 1
        for key, count in row["recovery_failure_reasons"].items():
            failure_reasons[str(key)] += int(count)

    return {
        "records": records,
        "success_rate": mean(float(row["success"]) for row in payloads),
        "raw_any_feasible_rate": mean(
            float(row["raw_any_feasible"]) for row in payloads
        ),
        "proposal_feasible_rate": (
            total_raw_feasible / total_proposals if total_proposals else None
        ),
        "mean_gap": _mean([row["gap_to_pool_best"] for row in payloads]),
        "gap_std": _std([row["gap_to_pool_best"] for row in payloads]),
        "mean_raw_gap": _mean(
            [row["raw_gap_to_pool_best"] for row in payloads]
        ),
        "raw_gap_std": _std(
            [row["raw_gap_to_pool_best"] for row in payloads]
        ),
        "recovery_invocation_rate": len(invoked) / records if records else None,
        "recovery_record_success_rate": (
            len(recovered_records) / len(invoked) if invoked else None
        ),
        "mean_released_services_when_invoked": _mean(
            [float(row["recovery_released_services"]) for row in invoked]
        ),
        "mean_completion_steps_when_invoked": _mean(
            [float(row["recovery_completion_steps"]) for row in invoked]
        ),
        "source_rates": {
            key: value / records for key, value in sorted(source_counts.items())
        },
        "recovery_failure_reasons": dict(sorted(failure_reasons.items())),
        "mean_sampling_seconds": _mean(
            [row["sampling_seconds"] for row in payloads]
        ),
        "mean_recovery_seconds": _mean(
            [row["recovery_seconds"] for row in payloads]
        ),
        "mean_total_seconds": _mean([row["total_seconds"] for row in payloads]),
    }


def _sign_test_p_value(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    if n <= 1024:
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        return min(1.0, 2.0 * tail)
    mu = n / 2.0
    sigma = math.sqrt(n / 4.0)
    z = (k + 0.5 - mu) / sigma
    return min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))


def _paired(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
) -> dict[str, Any]:
    wins = losses = ties = skipped = 0
    differences: list[float] = []
    for row in rows:
        left_gap = row["methods"][left]["gap_to_pool_best"]
        right_gap = row["methods"][right]["gap_to_pool_best"]
        if left_gap is None or right_gap is None:
            skipped += 1
            continue
        delta = float(right_gap) - float(left_gap)
        differences.append(delta)
        if delta > 1e-12:
            wins += 1
        elif delta < -1e-12:
            losses += 1
        else:
            ties += 1
    return {
        "left": left,
        "right": right,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "skipped": skipped,
        "mean_gap_reduction": _mean(differences),
        "p_value_two_sided_sign_test": _sign_test_p_value(wins, losses),
    }


def finalize(root: Path, output_root: Path) -> dict[str, Any]:
    rows = _records(output_root)
    if not rows:
        raise RuntimeError(f"No records found under {output_root / 'records'}")
    protocol = _read_json(output_root / "protocol.json")
    methods = tuple(protocol["methods"])
    by_setting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_partition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        setting = str(row["setting"])
        by_setting[setting].append(row)
        by_partition[(setting, str(row["partition"]))].append(row)

    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "protocol_sha256": _BASE._SEQUENTIAL._hash_payload(protocol),
        "methods": protocol["methods"],
        "method_labels": METHOD_LABELS,
        "overall": {
            method: _aggregate([row["methods"][method] for row in rows])
            for method in methods
        },
        "by_setting": {},
        "by_partition": {},
        "paired_main_evaluation": {},
    }
    for setting, setting_rows in sorted(by_setting.items()):
        evidence["by_setting"][setting] = {
            method: _aggregate([row["methods"][method] for row in setting_rows])
            for method in methods
        }
    for (setting, partition), partition_rows in sorted(by_partition.items()):
        evidence["by_partition"].setdefault(setting, {})[partition] = {
            method: _aggregate([row["methods"][method] for row in partition_rows])
            for method in methods
        }

    main_rows = by_setting.get("main_evaluation", [])
    for baseline in ("direct_b64_t1", "sequential_b64_t1"):
        evidence["paired_main_evaluation"][baseline] = _paired(
            main_rows,
            "masked_k8_t1",
            baseline,
        )

    evidence_path = output_root / "phase7_proposal_recovery_evidence.json"
    report_path = output_root / "phase7_proposal_recovery_report.md"
    freeze_path = output_root / "phase7_evidence_freeze.json"
    write_json(evidence_path, evidence)
    report_path.write_text(_report(evidence), encoding="utf-8")
    write_json(
        freeze_path,
        {
            "scope": f"{EVIDENCE_SCOPE}_freeze",
            "protocol": _relative(root, output_root / "protocol.json"),
            "protocol_sha256": file_sha256(output_root / "protocol.json"),
            "evidence": _relative(root, evidence_path),
            "evidence_sha256": file_sha256(evidence_path),
            "report": _relative(root, report_path),
            "report_sha256": file_sha256(report_path),
            "record_count": len(rows),
        },
    )
    return {
        "records": len(rows),
        "evidence": _relative(root, evidence_path),
        "report": _relative(root, report_path),
        "freeze": _relative(root, freeze_path),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _report(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 7 Proposal-Conditioned Recovery Evaluation",
        "",
        "All learned checkpoints and datasets are frozen. Raw proposals are verified",
        "first. Bounded recovery is invoked only when no raw proposal verifies, and",
        "every recovered placement is hard-verified before exact-latency selection.",
        "",
        f"Records: {evidence['records']}",
        "",
        "## By Setting",
        "",
    ]
    for setting, methods in evidence["by_setting"].items():
        lines.extend(
            [
                f"### {setting}",
                "",
                "| Method | Records | Success | Raw success | Proposal feasible | Gap | Raw gap | Recovery invoked | Recovery success | Time (s) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in methods.items():
            lines.append(
                f"| {METHOD_LABELS.get(method, method)} | {row['records']} | "
                f"{_pct(row['success_rate'])} | {_pct(row['raw_any_feasible_rate'])} | "
                f"{_pct(row['proposal_feasible_rate'])} | {_pct(row['mean_gap'])} | "
                f"{_pct(row['mean_raw_gap'])} | {_pct(row['recovery_invocation_rate'])} | "
                f"{_pct(row['recovery_record_success_rate'])} | "
                f"{_num(row['mean_total_seconds'])} |"
            )
        lines.append("")

    lines.extend(["## Main-Evaluation Paired Tests", ""])
    for baseline, row in evidence["paired_main_evaluation"].items():
        lines.append(
            f"- Masked Diffusion vs. {METHOD_LABELS[baseline]}: "
            f"wins/losses/ties/skipped={row['wins']}/{row['losses']}/"
            f"{row['ties']}/{row['skipped']}, mean Gap reduction="
            f"{_pct(row['mean_gap_reduction'])}, "
            f"two-sided sign-test p={_num(row['p_value_two_sided_sign_test'], 6)}."
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Gap is relative to the best verified MILP-pool objective for each instance.",
            "- Raw success measures the neural or random proposal set before recovery.",
            "- Recovery is proposal-conditioned, bounded, and may report failure.",
            "- This evaluation changes post-generation inference only; no model was retrained.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "finalize", "all"))
    parser.add_argument("--settings", default=",".join(DATASET_SPECS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--training-freeze",
        default="artifacts/phase6e-e-stage39-10seed-training/ten_seed_training_freeze.json",
    )
    parser.add_argument(
        "--sequential-checkpoint-root",
        default="artifacts/phase6f-sequential-conditional-training",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase7-proposal-conditioned-recovery",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-instances-per-partition", type=int)
    parser.add_argument("--no-skip-missing-datasets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = _parse_csv(args.settings, DATASET_SPECS)
    methods = _parse_csv(args.methods, DEFAULT_METHODS)
    unknown_settings = sorted(set(settings) - set(DATASET_SPECS))
    unknown_methods = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown_settings:
        raise ValueError(f"Unsupported settings: {unknown_settings}")
    if unknown_methods:
        raise ValueError(f"Unsupported methods: {unknown_methods}")
    if args.max_instances_per_partition is not None and args.max_instances_per_partition < 1:
        raise ValueError("--max-instances-per-partition must be positive.")

    output_root = _resolve(root, args.output_root)
    kwargs = {
        "settings": settings,
        "selected_seeds": _parse_seeds(args.seeds),
        "methods": methods,
        "training_freeze": _resolve(root, args.training_freeze),
        "sequential_checkpoint_root": _resolve(root, args.sequential_checkpoint_root),
        "output_root": output_root,
        "device_name": args.device,
        "max_instances_per_partition": args.max_instances_per_partition,
        "fallback_max_search_nodes": 1,
        "skip_missing_datasets": not bool(args.no_skip_missing_datasets),
    }
    _configure_base()
    if args.action in {"run", "all"}:
        print(_BASE.run_evaluation(root, **kwargs), flush=True)
    if args.action in {"finalize", "all"}:
        print(finalize(root, output_root), flush=True)


if __name__ == "__main__":
    main()
