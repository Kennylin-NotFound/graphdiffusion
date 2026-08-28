"""Single-seed held-out evaluation for the Sequential Conditional GNN baseline.

This script compares Direct K=64, Masked Diffusion K=8, and Sequential GNN
under the same no-repair verified-raw-then-fallback policy.  It is intentionally
separate from the ten-seed Phase 6F campaign so that pilot evidence cannot
overwrite frozen multi-seed artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.experiments.phase6ee_stage3 import load_stage3_solver
from gdm_factor_diffusion.experiments.schema import file_sha256
from gdm_factor_diffusion.graph import GraphFeatureSchema, build_factor_graph_batch
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    SequentialDecodeConfig,
    sample_direct_proposals,
    sample_masked_proposals,
    sample_sequential_proposals,
    solve_from_proposals,
)
from gdm_factor_diffusion.models import (
    SequentialPolicyConfig,
    TypedFactorSequentialPolicy,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze


SCOPE = "phase6f_sequential_single_seed_eval"
RECORD_SCOPE = f"{SCOPE}_record"
EVIDENCE_SCOPE = f"{SCOPE}_evidence"

DEFAULT_SEED = 2026070114
DIRECT = "direct_k64"
MASKED = "masked_diffusion_k8"
SEQUENTIAL = "sequential_kseq"
METHOD_ORDER = (DIRECT, MASKED, SEQUENTIAL)

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "sealed_id": {
        "dataset_root": "artifacts/datasets/phase6e-e-stage38-sealed",
        "partitions": ("sealed_test_id",),
    },
    "controlled_shift": {
        "dataset_root": "artifacts/datasets/phase6e-e-controlled-shift",
        "partitions": (
            "test_id_reference",
            "shift_high_sharing",
            "shift_low_compatibility",
            "shift_tight_capacity",
            "shift_unseen_workflow",
        ),
    },
    "realistic_profile": {
        "dataset_root": "artifacts/datasets/phase6e-e-realistic-profile",
        "partitions": ("profile_id", "profile_branched", "profile_high_sharing"),
    },
}


@dataclass(frozen=True, slots=True)
class SequentialSolver:
    model: torch.nn.Module
    feature_schema: GraphFeatureSchema
    model_kind: str


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(root: Path, path: str | Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest().upper()


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _parse_settings(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return tuple(DATASET_SPECS)
    settings = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(settings) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(f"Unsupported settings: {unknown}")
    return settings


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


def _feature_schema(payload: Mapping[str, Any]) -> GraphFeatureSchema:
    return GraphFeatureSchema(
        service_feature_names=tuple(payload["service_feature_names"]),
        device_feature_names=tuple(payload["device_feature_names"]),
        resource_names=tuple(payload["resource_names"]),
    )


def load_sequential_solver(
    checkpoint_path: str | Path,
    dataset: LabeledDeploymentDataset,
    device: torch.device | str,
) -> SequentialSolver:
    target = torch.device(device)
    payload = torch.load(Path(checkpoint_path), map_location=target, weights_only=True)
    if payload.get("model_kind") != "sequential_conditional":
        raise ValueError("Expected a sequential_conditional checkpoint.")
    metadata = payload["metadata"]
    schema = _feature_schema(metadata["feature_schema"])
    reference = build_factor_graph_batch(
        [dataset[0].instance],
        feature_schema=schema,
    ).to(target)
    config = SequentialPolicyConfig(**metadata["model_config"])
    model = TypedFactorSequentialPolicy.from_batch(reference, config).to(target)
    model.load_state_dict(payload["model"])
    model.eval()
    return SequentialSolver(model=model, feature_schema=schema, model_kind="sequential_conditional")


def _load_training_entry(
    root: Path,
    training: Mapping[str, Any],
    seed: int,
    kind: str,
) -> Path:
    return _resolve(root, training["runs"][str(seed)][kind]["paths"]["best.pt"])


def _protocol(
    *,
    root: Path,
    settings: tuple[str, ...],
    seed: int,
    training_freeze: Path,
    sequential_checkpoint: Path,
    output_root: Path,
    device: str,
    max_instances_per_partition: int | None,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scope": SCOPE,
        "settings": list(settings),
        "seed": int(seed),
        "training_freeze": _relative(root, training_freeze),
        "sequential_checkpoint": _relative(root, sequential_checkpoint),
        "output_root": _relative(root, output_root),
        "device": device,
        "deterministic": True,
        "evaluation_seed": 2026073011,
        "sample_batch_size": 8,
        "temperature": 1.0,
        "forward_equivalent_budget": 64,
        "max_instances_per_partition": max_instances_per_partition,
        "fallback_max_search_nodes": int(fallback_max_search_nodes),
        "policy": {
            "name": "verified_raw_then_fallback",
            "enable_repair": False,
            "enable_fallback": True,
            "always_include_fallback": False,
        },
        "methods": {
            DIRECT: {
                "family": "direct",
                "samples": 64,
                "forward_equivalent_budget": 64,
            },
            MASKED: {
                "family": "masked",
                "samples": 8,
                "stochastic": True,
                "neural_steps_per_proposal": 8,
                "forward_equivalent_budget": 64,
            },
            SEQUENTIAL: {
                "family": "sequential",
                "samples": "floor(64 / num_services)",
                "stochastic": True,
                "forward_equivalent_budget": "<=64",
            },
        },
        "dataset_specs": {
            name: {
                "dataset_root": spec["dataset_root"],
                "partitions": list(spec["partitions"]),
            }
            for name, spec in DATASET_SPECS.items()
            if name in settings
        },
    }


def _inference_config(protocol: Mapping[str, Any], samples: int) -> InferenceConfig:
    return InferenceConfig(
        num_samples=samples,
        sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
        fallback_max_search_nodes=int(protocol["fallback_max_search_nodes"]),
        enable_repair=False,
        enable_fallback=True,
        always_include_fallback=False,
    )


def _sequential_samples(instance: Any, protocol: Mapping[str, Any]) -> int:
    budget = int(protocol["forward_equivalent_budget"])
    return max(1, budget // max(1, int(instance.num_services)))


def _sample_method(
    *,
    method_id: str,
    method: Mapping[str, Any],
    direct_solver: Any,
    masked_solver: Any,
    sequential_solver: SequentialSolver,
    instance: Any,
    protocol: Mapping[str, Any],
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    family = method["family"]
    if family == "direct":
        samples = int(method["samples"])
        proposals, probabilities, seconds = sample_direct_proposals(
            direct_solver.model,
            instance,
            direct_solver.feature_schema,
            config=InferenceConfig(
                num_samples=samples,
                sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
            ),
            device=device,
            generator=generator,
        )
        return {
            "proposals": proposals,
            "probabilities": probabilities,
            "seconds": seconds,
            "realized_budget": samples,
        }
    if family == "masked":
        samples = int(method["samples"])
        sampled = sample_masked_proposals(
            masked_solver.model,
            instance,
            masked_solver.schedule,
            masked_solver.feature_schema,
            config=MaskedDecodeConfig(
                num_samples=samples,
                sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
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
            "completed_rate": float(sampled.completed.mean()),
            "realized_budget": samples * int(method["neural_steps_per_proposal"]),
        }
    if family == "sequential":
        samples = _sequential_samples(instance, protocol)
        sampled = sample_sequential_proposals(
            sequential_solver.model,
            instance,
            sequential_solver.feature_schema,
            config=SequentialDecodeConfig(
                num_samples=samples,
                sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
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
            "sequential_model_forwards": sampled.model_forwards,
            "completed_rate": float(sampled.completed.mean()),
            "realized_budget": samples * int(instance.num_services),
            "samples": samples,
        }
    raise ValueError(f"Unsupported method family: {family}")


def _payload(result: Any, pool_best: float, sampled: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metrics = result.metrics
    raw = metrics.get("best_raw_objective")
    payload = {
        "success": bool(result.success),
        "source": result.source,
        "objective": result.objective,
        "gap_to_pool_best": (
            None if result.objective is None else float(result.objective) / pool_best - 1.0
        ),
        "raw_success": raw is not None,
        "raw_gap_to_pool_best": None if raw is None else float(raw) / pool_best - 1.0,
        "raw_any_feasible": bool(metrics.get("raw_any_feasible", False)),
        "raw_feasible_count": int(metrics.get("raw_feasible_count", 0)),
        "num_raw_proposals": int(metrics.get("num_raw_proposals", 0)),
        "raw_feasible_rate": metrics.get("raw_feasible_rate"),
        "raw_unique_rate": metrics.get("raw_unique_rate"),
        "raw_pairwise_hamming": metrics.get("raw_pairwise_hamming"),
        "fallback_invoked": bool(metrics.get("fallback_invoked", False)),
        "fallback_success": bool(metrics.get("fallback_success", False)),
        "sampling_seconds": float(metrics.get("sampling_seconds", 0.0)),
        "verification_seconds": float(metrics.get("verification_seconds", 0.0)),
        "fallback_seconds": float(metrics.get("fallback_seconds", 0.0)),
        "total_seconds": float(metrics.get("total_seconds", 0.0)),
    }
    if sampled is not None:
        for key in (
            "realized_budget",
            "completed_rate",
            "masked_model_forwards",
            "sequential_model_forwards",
            "samples",
        ):
            if key in sampled:
                payload[key] = sampled[key]
    return payload


def _record_valid(
    path: Path,
    *,
    protocol_hash: str,
    training_freeze_sha256: str,
    sequential_checkpoint_sha256: str,
    dataset_freeze_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    row = _read_json(path)
    return (
        row.get("scope") == RECORD_SCOPE
        and row.get("protocol_sha256") == protocol_hash
        and row.get("training_freeze_sha256") == training_freeze_sha256
        and row.get("sequential_checkpoint_sha256") == sequential_checkpoint_sha256
        and row.get("dataset_freeze_sha256") == dataset_freeze_sha256
    )


def _run_setting(
    root: Path,
    *,
    setting: str,
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_hash: str,
    training: Mapping[str, Any],
    training_freeze: Path,
    sequential_checkpoint: Path,
    output_root: Path,
) -> dict[str, Any]:
    dataset_root = _resolve(root, spec["dataset_root"])
    freeze = audit_dataset_freeze(dataset_root)
    dataset_hash = file_sha256(dataset_root / "dataset_freeze.json")
    training_hash = file_sha256(training_freeze)
    sequential_hash = file_sha256(sequential_checkpoint)
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=tuple(spec["partitions"]),
        require_freeze=True,
    )
    indices = _limited_indices(dataset, protocol["max_instances_per_partition"])
    device = _device(str(protocol["device"]))
    seed = int(protocol["seed"])
    direct = load_stage3_solver(
        _load_training_entry(root, training, seed, "direct"),
        dataset,
        device,
    )
    masked = load_stage3_solver(
        _load_training_entry(root, training, seed, "masked_conditional"),
        dataset,
        device,
    )
    if masked.schedule is None:
        raise ValueError("Masked checkpoint has no absorbing-MASK schedule.")
    sequential = load_sequential_solver(sequential_checkpoint, dataset, device)

    completed = 0
    for index in indices:
        item = dataset[index]
        record_path = (
            output_root
            / "records"
            / setting
            / str(seed)
            / item.partition
            / f"{item.instance.instance_id}.json"
        )
        if _record_valid(
            record_path,
            protocol_hash=protocol_hash,
            training_freeze_sha256=training_hash,
            sequential_checkpoint_sha256=sequential_hash,
            dataset_freeze_sha256=dataset_hash,
        ):
            completed += 1
            continue
        pool_best = float(np.min(item.pool.latencies))
        method_results: dict[str, Any] = {}
        for method_id, method in protocol["methods"].items():
            sampled = _sample_method(
                method_id=method_id,
                method=method,
                direct_solver=direct,
                masked_solver=masked,
                sequential_solver=sequential,
                instance=item.instance,
                protocol=protocol,
                device=device,
                generator=torch.Generator(device=device).manual_seed(
                    derive_seed(
                        int(protocol["evaluation_seed"]),
                        (
                            f"{SCOPE}:{setting}:{method_id}:{seed}:"
                            f"{item.partition}:{item.instance.instance_id}"
                        ),
                    )
                ),
            )
            result = solve_from_proposals(
                item.instance,
                sampled["proposals"],
                model_probabilities=sampled["probabilities"],
                config=_inference_config(
                    protocol,
                    int(sampled.get("samples", method.get("samples", 1))),
                ),
                sampling_seconds=float(sampled["seconds"]),
                proposal_method=method_id,
            )
            method_results[method_id] = _payload(result, pool_best, sampled)
        write_json(
            record_path,
            {
                "schema_version": "1.0",
                "scope": RECORD_SCOPE,
                "setting": setting,
                "seed": seed,
                "instance_id": item.instance.instance_id,
                "partition": item.partition,
                "num_services": int(item.instance.num_services),
                "num_devices": int(item.instance.num_devices),
                "pool_best": pool_best,
                "pool_size": int(item.pool.size),
                "dataset_family": freeze.get("dataset_name"),
                "protocol_sha256": protocol_hash,
                "training_freeze_sha256": training_hash,
                "sequential_checkpoint_sha256": sequential_hash,
                "dataset_freeze_sha256": dataset_hash,
                "methods": method_results,
            },
        )
        completed += 1
    return {
        "setting": setting,
        "records_completed": completed,
        "expected": len(indices),
        "dataset_root": _relative(root, dataset_root),
        "partitions": list(spec["partitions"]),
    }


def run_evaluation(
    root: Path,
    *,
    settings: tuple[str, ...],
    seed: int,
    training_freeze: Path,
    sequential_checkpoint: Path,
    output_root: Path,
    device_name: str,
    max_instances_per_partition: int | None,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    protocol = _protocol(
        root=root,
        settings=settings,
        seed=seed,
        training_freeze=training_freeze,
        sequential_checkpoint=sequential_checkpoint,
        output_root=output_root,
        device=device_name,
        max_instances_per_partition=max_instances_per_partition,
        fallback_max_search_nodes=fallback_max_search_nodes,
    )
    protocol_hash = _hash_payload(protocol)
    seed_everything(int(protocol["evaluation_seed"]), deterministic=True)
    training = _read_json(training_freeze)
    if str(seed) not in training["runs"]:
        raise ValueError(f"Seed {seed} is absent from the training freeze.")
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "protocol.json", protocol)
    summaries = []
    for setting in settings:
        summaries.append(
            _run_setting(
                root,
                setting=setting,
                spec=DATASET_SPECS[setting],
                protocol=protocol,
                protocol_hash=protocol_hash,
                training=training,
                training_freeze=training_freeze,
                sequential_checkpoint=sequential_checkpoint,
                output_root=output_root,
            )
        )
    return {
        "scope": SCOPE,
        "protocol_sha256": protocol_hash,
        "summaries": summaries,
        "output_root": _relative(root, output_root),
    }


def _records(output_root: Path) -> list[Mapping[str, Any]]:
    return [_read_json(path) for path in sorted((output_root / "records").rglob("*.json"))]


def _finite(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _mean(values: list[float | None]) -> float | None:
    data = _finite(values)
    return mean(data) if data else None


def _std(values: list[float | None]) -> float | None:
    data = _finite(values)
    if not data:
        return None
    return pstdev(data) if len(data) > 1 else 0.0


def _aggregate(rows: list[Mapping[str, Any]], method_id: str) -> dict[str, Any]:
    payloads = [row["methods"][method_id] for row in rows]
    total_proposals = sum(int(payload["num_raw_proposals"]) for payload in payloads)
    total_feasible = sum(int(payload["raw_feasible_count"]) for payload in payloads)
    sources = defaultdict(int)
    for payload in payloads:
        sources[str(payload["source"])] += 1
    return {
        "records": len(payloads),
        "success_rate": mean(float(payload["success"]) for payload in payloads),
        "raw_any_feasible_rate": mean(float(payload["raw_any_feasible"]) for payload in payloads),
        "proposal_feasible_rate": total_feasible / total_proposals if total_proposals else None,
        "mean_raw_gap": _mean([payload["raw_gap_to_pool_best"] for payload in payloads]),
        "raw_gap_std": _std([payload["raw_gap_to_pool_best"] for payload in payloads]),
        "mean_final_gap": _mean([payload["gap_to_pool_best"] for payload in payloads]),
        "fallback_invocation_rate": mean(float(payload["fallback_invoked"]) for payload in payloads),
        "mean_unique_rate": _mean([payload.get("raw_unique_rate") for payload in payloads]),
        "mean_pairwise_hamming": _mean([payload.get("raw_pairwise_hamming") for payload in payloads]),
        "mean_completed_rate": _mean([payload.get("completed_rate") for payload in payloads]),
        "mean_realized_budget": _mean([payload.get("realized_budget") for payload in payloads]),
        "mean_total_seconds": _mean([payload["total_seconds"] for payload in payloads]),
        "source_rates": {source: count / len(payloads) for source, count in sources.items()},
    }


def _sign_p_value(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    if n <= 1024:
        return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))
    mu = n / 2.0
    sigma = math.sqrt(n / 4.0)
    z = (k + 0.5 - mu) / sigma
    return min(1.0, 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0)))


def _paired(rows: list[Mapping[str, Any]], left: str, right: str, *, stage: str) -> dict[str, Any]:
    wins = losses = ties = skipped = 0
    key = "raw_gap_to_pool_best" if stage == "raw" else "gap_to_pool_best"
    for row in rows:
        left_value = row["methods"][left][key]
        right_value = row["methods"][right][key]
        if left_value is None or right_value is None:
            skipped += 1
            continue
        if float(left_value) < float(right_value) - 1e-12:
            wins += 1
        elif float(right_value) < float(left_value) - 1e-12:
            losses += 1
        else:
            ties += 1
    return {
        "left": left,
        "right": right,
        "stage": stage,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "skipped": skipped,
        "p_value_two_sided_sign_test": _sign_p_value(wins, losses),
    }


def finalize(root: Path, *, output_root: Path) -> dict[str, Any]:
    rows = _records(output_root)
    if not rows:
        raise RuntimeError(f"No records found under {output_root / 'records'}")
    protocol = _read_json(output_root / "protocol.json")
    by_setting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setting[str(row["setting"])].append(row)
    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "protocol": protocol,
        "overall": {method: _aggregate(rows, method) for method in METHOD_ORDER},
        "by_setting": {},
        "paired": {
            "sequential_vs_masked_raw": _paired(rows, SEQUENTIAL, MASKED, stage="raw"),
            "sequential_vs_masked_final": _paired(rows, SEQUENTIAL, MASKED, stage="final"),
            "sequential_vs_direct_raw": _paired(rows, SEQUENTIAL, DIRECT, stage="raw"),
            "sequential_vs_direct_final": _paired(rows, SEQUENTIAL, DIRECT, stage="final"),
        },
    }
    for setting, setting_rows in sorted(by_setting.items()):
        evidence["by_setting"][setting] = {
            method: _aggregate(setting_rows, method) for method in METHOD_ORDER
        }
    evidence_path = output_root / "sequential_single_seed_evidence.json"
    report_path = output_root / "sequential_single_seed_report.md"
    write_json(evidence_path, evidence)
    report_path.write_text(_report(evidence), encoding="utf-8")
    return {
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "evidence": _relative(root, evidence_path),
        "report": _relative(root, report_path),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _row(method_id: str, values: Mapping[str, Any]) -> str:
    return (
        f"| {method_id} | {_pct(values['success_rate'])} | "
        f"{_pct(values['raw_any_feasible_rate'])} | "
        f"{_pct(values['proposal_feasible_rate'])} | "
        f"{_pct(values['mean_raw_gap'])} | "
        f"{_pct(values['mean_final_gap'])} | "
        f"{_pct(values['fallback_invocation_rate'])} | "
        f"{_pct(values['mean_completed_rate'])} | "
        f"{_num(values['mean_realized_budget'])} | "
        f"{_num(values['mean_total_seconds'])} s |"
    )


def _paired_line(name: str, values: Mapping[str, Any]) -> str:
    return (
        f"- {name}: wins/losses/ties/skipped="
        f"{values['wins']}/{values['losses']}/{values['ties']}/{values['skipped']}, "
        f"p={_num(values['p_value_two_sided_sign_test'], 6)}."
    )


def _report(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Sequential Conditional GNN Single-Seed Held-Out Evaluation",
        "",
        "Policy: no repair; verified raw proposals are selected by exact latency,",
        "and fallback is invoked only when no raw proposal is feasible.",
        "",
        f"Records: {evidence['records']}",
        "",
        "## Overall",
        "",
        "| Method | Success | Raw any feasible | Proposal feasible | Raw gap | Final gap | Fallback | Completed | B_NN | Time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        lines.append(_row(method, evidence["overall"][method]))
    lines.extend(["", "## Paired Tests", ""])
    for name, values in evidence["paired"].items():
        lines.append(_paired_line(name, values))
    lines.extend(["", "## By Setting", ""])
    for setting, methods in evidence["by_setting"].items():
        lines.extend(
            [
                f"### {setting}",
                "",
                "| Method | Success | Raw any feasible | Proposal feasible | Raw gap | Final gap | Fallback | Completed | B_NN | Time |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHOD_ORDER:
            lines.append(_row(method, methods[method]))
        lines.append("")
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "- This is a single-seed diagnostic evaluation, not multi-seed manuscript evidence.",
            "- Use it to decide whether Sequential GNN deserves a formal multi-seed campaign.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "finalize", "all"))
    parser.add_argument("--settings", default=",".join(DATASET_SPECS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--training-freeze",
        default=(
            "artifacts/phase6e-e-stage39-10seed-training/"
            "ten_seed_training_freeze.json"
        ),
    )
    parser.add_argument(
        "--sequential-checkpoint",
        default=(
            "artifacts/phase6f-sequential-conditional-training/"
            "sequential_conditional-seed2026070114/best.pt"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase6f-sequential-single-seed-evaluation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-instances-per-partition", type=int, default=None)
    parser.add_argument("--fallback-max-search-nodes", type=int, default=30000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = _parse_settings(args.settings)
    output_root = _resolve(root, args.output_root)
    kwargs = {
        "settings": settings,
        "seed": int(args.seed),
        "training_freeze": _resolve(root, args.training_freeze),
        "sequential_checkpoint": _resolve(root, args.sequential_checkpoint),
        "output_root": output_root,
        "device_name": args.device,
        "max_instances_per_partition": args.max_instances_per_partition,
        "fallback_max_search_nodes": int(args.fallback_max_search_nodes),
    }
    if args.action in {"run", "all"}:
        print(run_evaluation(root, **kwargs), flush=True)
    if args.action in {"finalize", "all"}:
        print(finalize(root, output_root=output_root), flush=True)


if __name__ == "__main__":
    main()
