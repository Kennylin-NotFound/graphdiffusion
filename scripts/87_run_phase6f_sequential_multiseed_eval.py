"""Multi-seed Sequential GNN evaluation under the Phase 6F no-repair policy.

The script evaluates only the Sequential Conditional GNN.  During finalization
it reads the already completed Phase 6F Direct/Masked records and compares the
matched seed-instance outputs under the same verified-raw-then-fallback policy.
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
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.experiments.schema import file_sha256
from gdm_factor_diffusion.graph import GraphFeatureSchema, build_factor_graph_batch
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    SequentialDecodeConfig,
    sample_sequential_proposals,
    solve_from_proposals,
)
from gdm_factor_diffusion.models import (
    SequentialPolicyConfig,
    TypedFactorSequentialPolicy,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze


SCOPE = "phase6f_sequential_multiseed_eval"
RECORD_SCOPE = f"{SCOPE}_record"
EVIDENCE_SCOPE = f"{SCOPE}_evidence"

DEFAULT_SEEDS = tuple(range(2026070111, 2026070121))
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


def _parse_csv(value: str | None, *, default: Sequence[str]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return tuple(default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_seeds(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return DEFAULT_SEEDS
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    unknown = sorted(set(seeds) - set(DEFAULT_SEEDS))
    if unknown:
        raise ValueError(f"Unsupported seeds: {unknown}")
    return seeds


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


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


def _sequential_checkpoint(root: Path, checkpoint_root: Path, seed: int) -> Path:
    path = checkpoint_root / f"sequential_conditional-seed{seed}" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Sequential checkpoint for seed {seed}: {path}")
    return path


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
    reference = build_factor_graph_batch([dataset[0].instance], feature_schema=schema).to(target)
    model = TypedFactorSequentialPolicy.from_batch(
        reference,
        SequentialPolicyConfig(**metadata["model_config"]),
    ).to(target)
    model.load_state_dict(payload["model"])
    model.eval()
    return SequentialSolver(model=model, feature_schema=schema)


def _protocol(
    *,
    root: Path,
    settings: tuple[str, ...],
    selected_seeds: tuple[int, ...],
    checkpoint_root: Path,
    output_root: Path,
    device: str,
    max_instances_per_partition: int | None,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "scope": SCOPE,
        "settings": list(settings),
        "selected_seeds": list(selected_seeds),
        "sequential_checkpoint_root": _relative(root, checkpoint_root),
        "output_root": _relative(root, output_root),
        "device": device,
        "deterministic": True,
        "evaluation_seed": 2026071011,
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
            SEQUENTIAL: {
                "family": "sequential",
                "samples": "floor(64 / num_services)",
                "stochastic": True,
                "forward_equivalent_budget": "<=64",
            }
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
    return max(1, int(protocol["forward_equivalent_budget"]) // max(1, int(instance.num_services)))


def _payload(result: Any, pool_best: float, sampled: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result.metrics
    raw = metrics.get("best_raw_objective")
    return {
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
        "completed_rate": float(sampled["completed_rate"]),
        "samples": int(sampled["samples"]),
        "realized_budget": int(sampled["realized_budget"]),
        "sequential_model_forwards": int(sampled["sequential_model_forwards"]),
    }


def _record_valid(
    path: Path,
    *,
    protocol_hash: str,
    checkpoint_sha256: str,
    dataset_freeze_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    row = _read_json(path)
    return (
        row.get("scope") == RECORD_SCOPE
        and row.get("protocol_sha256") == protocol_hash
        and row.get("sequential_checkpoint_sha256") == checkpoint_sha256
        and row.get("dataset_freeze_sha256") == dataset_freeze_sha256
    )


def _run_setting_seed(
    root: Path,
    *,
    setting: str,
    spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_hash: str,
    seed: int,
    checkpoint_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    dataset_root = _resolve(root, spec["dataset_root"])
    freeze = audit_dataset_freeze(dataset_root)
    dataset_hash = file_sha256(dataset_root / "dataset_freeze.json")
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=tuple(spec["partitions"]),
        require_freeze=True,
    )
    checkpoint = _sequential_checkpoint(root, checkpoint_root, seed)
    checkpoint_hash = file_sha256(checkpoint)
    device = _device(str(protocol["device"]))
    solver = load_sequential_solver(checkpoint, dataset, device)
    indices = _limited_indices(dataset, protocol["max_instances_per_partition"])
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
            checkpoint_sha256=checkpoint_hash,
            dataset_freeze_sha256=dataset_hash,
        ):
            completed += 1
            continue
        samples = _sequential_samples(item.instance, protocol)
        generator = torch.Generator(device=device).manual_seed(
            derive_seed(
                int(protocol["evaluation_seed"]),
                (
                    f"{SCOPE}:{setting}:{SEQUENTIAL}:{seed}:"
                    f"{item.partition}:{item.instance.instance_id}"
                ),
            )
        )
        sampled_batch = sample_sequential_proposals(
            solver.model,
            item.instance,
            solver.feature_schema,
            config=SequentialDecodeConfig(
                num_samples=samples,
                sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
                stochastic=True,
                temperature=float(protocol["temperature"]),
            ),
            device=device,
            generator=generator,
        )
        sampled = {
            "proposals": sampled_batch.proposals,
            "probabilities": sampled_batch.probabilities,
            "seconds": sampled_batch.sampling_seconds,
            "completed_rate": float(sampled_batch.completed.mean()),
            "samples": samples,
            "realized_budget": samples * int(item.instance.num_services),
            "sequential_model_forwards": sampled_batch.model_forwards,
        }
        result = solve_from_proposals(
            item.instance,
            sampled["proposals"],
            model_probabilities=sampled["probabilities"],
            config=_inference_config(protocol, samples),
            sampling_seconds=float(sampled["seconds"]),
            proposal_method=SEQUENTIAL,
        )
        write_json(
            record_path,
            {
                "schema_version": "1.0",
                "scope": RECORD_SCOPE,
                "setting": setting,
                "seed": int(seed),
                "instance_id": item.instance.instance_id,
                "partition": item.partition,
                "num_services": int(item.instance.num_services),
                "num_devices": int(item.instance.num_devices),
                "pool_best": float(np.min(item.pool.latencies)),
                "pool_size": int(item.pool.size),
                "dataset_family": freeze.get("dataset_name"),
                "protocol_sha256": protocol_hash,
                "sequential_checkpoint": _relative(root, checkpoint),
                "sequential_checkpoint_sha256": checkpoint_hash,
                "dataset_freeze_sha256": dataset_hash,
                "methods": {SEQUENTIAL: _payload(result, float(np.min(item.pool.latencies)), sampled)},
            },
        )
        completed += 1
    return {
        "setting": setting,
        "seed": int(seed),
        "records_completed": completed,
        "expected": len(indices),
    }


def run_evaluation(
    root: Path,
    *,
    settings: tuple[str, ...],
    selected_seeds: tuple[int, ...],
    checkpoint_root: Path,
    output_root: Path,
    device_name: str,
    max_instances_per_partition: int | None,
    fallback_max_search_nodes: int,
) -> dict[str, Any]:
    protocol = _protocol(
        root=root,
        settings=settings,
        selected_seeds=selected_seeds,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        device=device_name,
        max_instances_per_partition=max_instances_per_partition,
        fallback_max_search_nodes=fallback_max_search_nodes,
    )
    protocol_hash = _hash_payload(protocol)
    seed_everything(int(protocol["evaluation_seed"]), deterministic=True)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "protocol.json", protocol)
    summaries = []
    for seed in selected_seeds:
        for setting in settings:
            summaries.append(
                _run_setting_seed(
                    root,
                    setting=setting,
                    spec=DATASET_SPECS[setting],
                    protocol=protocol,
                    protocol_hash=protocol_hash,
                    seed=seed,
                    checkpoint_root=checkpoint_root,
                    output_root=output_root,
                )
            )
    return {
        "scope": SCOPE,
        "protocol_sha256": protocol_hash,
        "output_root": _relative(root, output_root),
        "summaries": summaries,
    }


def _records(root: Path) -> list[Mapping[str, Any]]:
    return [_read_json(path) for path in sorted((root / "records").rglob("*.json"))]


def _phase6f_record_index(records_root: Path) -> dict[tuple[str, int, str, str], Mapping[str, Any]]:
    rows = [_read_json(path) for path in sorted(records_root.rglob("*.json"))]
    return {
        (
            str(row["setting"]),
            int(row["training_seed"]),
            str(row["partition"]),
            str(row["instance_id"]),
        ): row
        for row in rows
    }


def _finite(values: Sequence[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _mean(values: Sequence[float | None]) -> float | None:
    data = _finite(values)
    return mean(data) if data else None


def _std(values: Sequence[float | None]) -> float | None:
    data = _finite(values)
    if not data:
        return None
    return pstdev(data) if len(data) > 1 else 0.0


def _aggregate_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_proposals = sum(int(payload["num_raw_proposals"]) for payload in payloads)
    total_feasible = sum(int(payload["raw_feasible_count"]) for payload in payloads)
    return {
        "records": len(payloads),
        "success_rate": mean(float(payload["success"]) for payload in payloads),
        "raw_any_feasible_rate": mean(float(payload["raw_any_feasible"]) for payload in payloads),
        "proposal_feasible_rate": total_feasible / total_proposals if total_proposals else None,
        "mean_raw_gap": _mean([payload["raw_gap_to_pool_best"] for payload in payloads]),
        "raw_gap_std": _std([payload["raw_gap_to_pool_best"] for payload in payloads]),
        "mean_final_gap": _mean([payload["gap_to_pool_best"] for payload in payloads]),
        "final_gap_std": _std([payload["gap_to_pool_best"] for payload in payloads]),
        "fallback_invocation_rate": mean(float(payload["fallback_invoked"]) for payload in payloads),
        "mean_unique_rate": _mean([payload.get("raw_unique_rate") for payload in payloads]),
        "mean_pairwise_hamming": _mean([payload.get("raw_pairwise_hamming") for payload in payloads]),
        "mean_completed_rate": _mean([payload.get("completed_rate") for payload in payloads]),
        "mean_realized_budget": _mean([payload.get("realized_budget") for payload in payloads]),
        "mean_total_seconds": _mean([payload["total_seconds"] for payload in payloads]),
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


def _paired(
    rows: Sequence[Mapping[str, Any]],
    phase_index: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    right_method: str,
    *,
    stage: str,
) -> dict[str, Any]:
    wins = losses = ties = skipped = missing = 0
    key = "raw_gap_to_pool_best" if stage == "raw" else "gap_to_pool_best"
    for row in rows:
        pair_key = (
            str(row["setting"]),
            int(row["seed"]),
            str(row["partition"]),
            str(row["instance_id"]),
        )
        phase = phase_index.get(pair_key)
        if phase is None:
            missing += 1
            continue
        left = row["methods"][SEQUENTIAL][key]
        right = phase["methods"][right_method]["verify_fallback"][key]
        if left is None or right is None:
            skipped += 1
            continue
        if float(left) < float(right) - 1e-12:
            wins += 1
        elif float(right) < float(left) - 1e-12:
            losses += 1
        else:
            ties += 1
    return {
        "left": SEQUENTIAL,
        "right": right_method,
        "stage": stage,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "skipped": skipped,
        "missing_phase6f_records": missing,
        "p_value_two_sided_sign_test": _sign_p_value(wins, losses),
    }


def finalize(
    root: Path,
    *,
    output_root: Path,
    phase6f_records_root: Path,
) -> dict[str, Any]:
    rows = _records(output_root)
    if not rows:
        raise RuntimeError(f"No Sequential records found under {output_root / 'records'}")
    phase_index = _phase6f_record_index(phase6f_records_root)
    matched_phase_payloads: dict[str, list[Mapping[str, Any]]] = {DIRECT: [], MASKED: []}
    sequential_payloads = [row["methods"][SEQUENTIAL] for row in rows]
    for row in rows:
        key = (
            str(row["setting"]),
            int(row["seed"]),
            str(row["partition"]),
            str(row["instance_id"]),
        )
        phase = phase_index.get(key)
        if phase is None:
            continue
        matched_phase_payloads[DIRECT].append(phase["methods"][DIRECT]["verify_fallback"])
        matched_phase_payloads[MASKED].append(phase["methods"][MASKED]["verify_fallback"])

    by_setting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setting[str(row["setting"])].append(row)

    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "matched_phase6f_records": len(matched_phase_payloads[MASKED]),
        "overall": {
            SEQUENTIAL: _aggregate_payloads(sequential_payloads),
            DIRECT: _aggregate_payloads(matched_phase_payloads[DIRECT]),
            MASKED: _aggregate_payloads(matched_phase_payloads[MASKED]),
        },
        "by_setting": {},
        "paired": {
            "sequential_vs_masked_raw": _paired(rows, phase_index, MASKED, stage="raw"),
            "sequential_vs_masked_final": _paired(rows, phase_index, MASKED, stage="final"),
            "sequential_vs_direct_raw": _paired(rows, phase_index, DIRECT, stage="raw"),
            "sequential_vs_direct_final": _paired(rows, phase_index, DIRECT, stage="final"),
        },
    }
    for setting, setting_rows in sorted(by_setting.items()):
        keys = [
            (
                str(row["setting"]),
                int(row["seed"]),
                str(row["partition"]),
                str(row["instance_id"]),
            )
            for row in setting_rows
        ]
        phase_payloads = {
            method: [
                phase_index[key]["methods"][method]["verify_fallback"]
                for key in keys
                if key in phase_index
            ]
            for method in (DIRECT, MASKED)
        }
        evidence["by_setting"][setting] = {
            SEQUENTIAL: _aggregate_payloads([row["methods"][SEQUENTIAL] for row in setting_rows]),
            DIRECT: _aggregate_payloads(phase_payloads[DIRECT]),
            MASKED: _aggregate_payloads(phase_payloads[MASKED]),
        }
    evidence_path = output_root / "sequential_multiseed_evidence.json"
    report_path = output_root / "sequential_multiseed_vs_masked_report.md"
    write_json(evidence_path, evidence)
    report_path.write_text(_report(evidence), encoding="utf-8")
    return {
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "matched_phase6f_records": len(matched_phase_payloads[MASKED]),
        "evidence": _relative(root, evidence_path),
        "report": _relative(root, report_path),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.3f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _row(method: str, values: Mapping[str, Any]) -> str:
    return (
        f"| {method} | {_pct(values['success_rate'])} | "
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
        f"- {name}: wins/losses/ties/skipped/missing="
        f"{values['wins']}/{values['losses']}/{values['ties']}/"
        f"{values['skipped']}/{values['missing_phase6f_records']}, "
        f"p={_num(values['p_value_two_sided_sign_test'], 6)}."
    )


def _report(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Sequential GNN Multi-Seed Evaluation vs Direct and Masked Diffusion",
        "",
        "Policy: no repair; verified raw proposals are selected by exact latency,",
        "and fallback is invoked only when no raw proposal is feasible.",
        "",
        f"Sequential records: {evidence['records']}",
        f"Matched Phase 6F records: {evidence['matched_phase6f_records']}",
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
            "- Sequential GNN is compared on the same seed-instance keys as Phase 6F.",
            "- Results are suitable for deciding whether to include Sequential GNN as a strong neural baseline.",
            "- Manuscript claims still need conservative wording if Sequential is competitive with Masked Diffusion.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "finalize", "all"))
    parser.add_argument("--settings", default=",".join(DATASET_SPECS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument(
        "--checkpoint-root",
        default="artifacts/phase6f-sequential-conditional-training",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase6f-sequential-multiseed-evaluation",
    )
    parser.add_argument(
        "--phase6f-records-root",
        default="artifacts/phase6f-no-repair-neural-proposals/records",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-instances-per-partition", type=int, default=None)
    parser.add_argument("--fallback-max-search-nodes", type=int, default=30000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = _resolve(root, args.output_root)
    if args.action in {"run", "all"}:
        print(
            run_evaluation(
                root,
                settings=_parse_csv(args.settings, default=DATASET_SPECS),
                selected_seeds=_parse_seeds(args.seeds),
                checkpoint_root=_resolve(root, args.checkpoint_root),
                output_root=output_root,
                device_name=args.device,
                max_instances_per_partition=args.max_instances_per_partition,
                fallback_max_search_nodes=int(args.fallback_max_search_nodes),
            ),
            flush=True,
        )
    if args.action in {"finalize", "all"}:
        print(
            finalize(
                root,
                output_root=output_root,
                phase6f_records_root=_resolve(root, args.phase6f_records_root),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
