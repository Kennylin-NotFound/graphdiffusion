"""Probe inference-time neural decoding enhancements.

This script is intentionally exploratory.  It evaluates whether stronger
proposal budgets and temperature settings can improve learned proposal
generators under the current verified-candidate policy:

1. generate learned proposals;
2. keep only hard-verified candidates;
3. select the verified candidate with minimum exact latency;
4. invoke fallback only if no verified learned candidate exists.

The latency-aware heuristic is intentionally excluded.  Greedy is kept only as
a simple single-pass deterministic reference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.diffusion import masked_softmax, sample_categorical
from gdm_factor_diffusion.experiments.phase6ee_stage3 import load_stage3_solver
from gdm_factor_diffusion.experiments.schema import file_sha256
from gdm_factor_diffusion.graph import build_factor_graph_batch
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    SequentialDecodeConfig,
    sample_masked_proposals,
    sample_sequential_proposals,
    solve_from_proposals,
    solve_greedy_local,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset, audit_dataset_freeze


SCOPE = "phase6f_decoding_enhancement_probe"
RECORD_SCOPE = f"{SCOPE}_record"
EVIDENCE_SCOPE = f"{SCOPE}_evidence"

DEFAULT_SEEDS = (2026070113,)
DEFAULT_METHODS = (
    "greedy",
    "direct_b64_t1",
    "direct_b64_t1_mixens",
    "sequential_b64_t1",
    "sequential_b64_t1_mixens",
    "masked_k8_t1",
    "masked_k8_t1_mix",
    "masked_k8_t1_ens",
    "masked_k8_t1_mixens",
)

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
    "cross_scale": {
        "dataset_root": "artifacts/datasets/phase6c-final-scale",
        "partitions": ("scale_medium", "scale_large", "scale_extra_large"),
    },
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(root: Path, path: str | Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_sequential_module() -> Any:
    path = Path(__file__).with_name("87_run_phase6f_sequential_multiseed_eval.py")
    spec = importlib.util.spec_from_file_location("phase6f_sequential_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Sequential helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SEQUENTIAL = _load_sequential_module()


def _parse_csv(value: str | None, *, default: Sequence[str]) -> tuple[str, ...]:
    if value is None or not value.strip():
        return tuple(default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_seeds(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return DEFAULT_SEEDS
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _method_spec(method_id: str) -> dict[str, Any]:
    if method_id == "greedy":
        return {"family": "greedy", "budget": 0, "samples": 1, "temperature": None}
    parts = method_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unsupported method id: {method_id!r}")
    family = parts[0]
    budget_part = parts[1]
    temp_part = parts[2]
    policy = parts[3] if len(parts) > 3 else "plain"
    if policy not in {"plain", "mix", "ens", "mixens"}:
        raise ValueError(f"Unsupported decoding policy in {method_id!r}: {policy!r}")
    if not budget_part.startswith(("b", "k")) or not temp_part.startswith("t"):
        raise ValueError(f"Unsupported method id: {method_id!r}")
    numeric = int(budget_part[1:])
    temperature_token = temp_part[1:]
    if "p" in temperature_token:
        temperature = float(temperature_token.replace("p", "."))
    elif temperature_token.startswith("0") and len(temperature_token) > 1:
        temperature = float("0." + temperature_token[1:])
    else:
        temperature = float(temperature_token)
    if family == "direct":
        samples = numeric
        budget = numeric
    elif family == "sequential":
        samples = None
        budget = numeric
    elif family == "masked":
        samples = numeric
        budget = numeric * 8
    else:
        raise ValueError(f"Unsupported method family: {family!r}")
    return {
        "family": family,
        "budget": budget,
        "samples": samples,
        "temperature": temperature,
        "policy": policy,
        "stochastic": True,
    }


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
        "evaluation_seed": 2026071511,
        "sample_batch_size": 8,
        "max_instances_per_partition": max_instances_per_partition,
        "fallback_max_search_nodes": int(fallback_max_search_nodes),
        "skip_missing_datasets": bool(skip_missing_datasets),
        "policy": {
            "name": "verified_learned_candidates_then_fallback",
            "enable_repair": False,
            "enable_fallback": True,
            "always_include_fallback": False,
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


def _inference_config(
    protocol: Mapping[str, Any],
    samples: int,
) -> InferenceConfig:
    return InferenceConfig(
        num_samples=samples,
        sample_batch_size=min(int(protocol["sample_batch_size"]), samples),
        fallback_max_search_nodes=int(protocol["fallback_max_search_nodes"]),
        enable_repair=False,
        enable_fallback=True,
        always_include_fallback=False,
    )


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


def _split_counts(total: int, weights: Sequence[float]) -> list[int]:
    if total < 0:
        raise ValueError("total must be nonnegative.")
    if total == 0:
        return [0 for _ in weights]
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        raise ValueError("weights must have positive sum.")
    raw = [total * float(weight) / weight_sum for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def _combine_sampled(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    active = [chunk for chunk in chunks if int(chunk["proposals"].shape[0]) > 0]
    if not active:
        raise ValueError("At least one sampled chunk is required.")
    total = sum(int(chunk["proposals"].shape[0]) for chunk in active)
    completed = [
        float(chunk.get("completed_rate", 1.0)) * int(chunk["proposals"].shape[0])
        for chunk in active
    ]
    realized = [
        float(chunk.get("realized_budget", chunk.get("model_forwards", 0.0)))
        for chunk in active
        if chunk.get("realized_budget") is not None
    ]
    return {
        "proposals": np.concatenate([chunk["proposals"] for chunk in active], axis=0),
        "probabilities": np.concatenate(
            [chunk["probabilities"] for chunk in active], axis=0
        ),
        "seconds": sum(float(chunk.get("seconds", 0.0)) for chunk in active),
        "model_forwards": sum(
            int(chunk.get("model_forwards", 0)) for chunk in active
        ),
        "completed_rate": sum(completed) / total,
        "realized_budget": sum(realized) if realized else None,
    }


@torch.no_grad()
def _sample_direct_temperature(
    solver: Any,
    instance: Any,
    *,
    samples: int,
    temperature: float,
    stochastic: bool,
    sample_batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    solver.model.eval()
    proposals: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    start = perf_counter()
    remaining = samples
    while remaining:
        chunk = min(sample_batch_size, remaining)
        batch = build_factor_graph_batch(
            [instance] * chunk,
            feature_schema=solver.feature_schema,
        ).to(device)
        logits = solver.model(batch) / float(temperature)
        probability = masked_softmax(logits, batch.candidate_mask, batch.service_mask)
        if stochastic:
            state = sample_categorical(
                probability,
                batch.candidate_mask,
                batch.service_mask,
                generator=generator,
            )
        else:
            state = probability.argmax(dim=-1).masked_fill(~batch.service_mask, -1)
        proposals.append(state[:, : instance.num_services].cpu().numpy())
        probabilities.append(
            probability[:, : instance.num_services, : instance.num_devices]
            .cpu()
            .numpy()
        )
        remaining -= chunk
    return {
        "proposals": np.concatenate(proposals, axis=0),
        "probabilities": np.concatenate(probabilities, axis=0),
        "seconds": perf_counter() - start,
        "model_forwards": samples,
        "completed_rate": 1.0,
        "realized_budget": samples,
    }


def _direct_chunk(
    solver: Any,
    instance: Any,
    *,
    samples: int,
    temperature: float,
    stochastic: bool,
    sample_batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any] | None:
    if samples <= 0:
        return None
    return _sample_direct_temperature(
        solver,
        instance,
        samples=samples,
        temperature=temperature,
        stochastic=stochastic,
        sample_batch_size=min(sample_batch_size, samples),
        device=device,
        generator=generator,
    )


def _masked_chunk(
    solver: Any,
    instance: Any,
    *,
    samples: int,
    temperature: float,
    stochastic: bool,
    sample_batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any] | None:
    if samples <= 0:
        return None
    sampled = sample_masked_proposals(
        solver.model,
        instance,
        solver.schedule,
        solver.feature_schema,
        config=MaskedDecodeConfig(
            num_samples=samples,
            sample_batch_size=min(sample_batch_size, samples),
            stochastic=stochastic,
            temperature=temperature,
        ),
        device=device,
        generator=generator,
    )
    return {
        "proposals": sampled.proposals,
        "probabilities": sampled.probabilities,
        "seconds": sampled.sampling_seconds,
        "model_forwards": sampled.model_forwards,
        "completed_rate": float(sampled.completed.mean()),
        "realized_budget": sampled.model_forwards,
    }


def _sequential_chunk(
    solver: Any,
    instance: Any,
    *,
    samples: int,
    temperature: float,
    stochastic: bool,
    sample_batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any] | None:
    if samples <= 0:
        return None
    sampled = sample_sequential_proposals(
        solver.model,
        instance,
        solver.feature_schema,
        config=SequentialDecodeConfig(
            num_samples=samples,
            sample_batch_size=min(sample_batch_size, samples),
            stochastic=stochastic,
            temperature=temperature,
        ),
        device=device,
        generator=generator,
    )
    return {
        "proposals": sampled.proposals,
        "probabilities": sampled.probabilities,
        "seconds": sampled.sampling_seconds,
        "model_forwards": sampled.model_forwards,
        "completed_rate": float(sampled.completed.mean()),
        "realized_budget": samples * int(instance.num_services),
    }


def _sample_method(
    *,
    method_id: str,
    spec: Mapping[str, Any],
    direct_solver: Any,
    masked_solver: Any,
    sequential_solver: Any,
    instance: Any,
    protocol: Mapping[str, Any],
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, Any]:
    family = spec["family"]
    sample_batch_size = int(protocol["sample_batch_size"])
    policy = str(spec.get("policy", "plain"))
    base_temperature = float(spec["temperature"])
    ensemble_temperatures = (0.75, 1.0, 1.25)
    ensemble_weights = (1.0, 2.0, 1.0)

    def compose(
        *,
        total_samples: int,
        sampler: Any,
    ) -> dict[str, Any]:
        if total_samples < 1:
            raise ValueError("total_samples must be positive.")
        chunks: list[dict[str, Any]] = []
        if policy == "plain":
            chunk = sampler(
                samples=total_samples,
                temperature=base_temperature,
                stochastic=True,
            )
            if chunk is not None:
                chunks.append(chunk)
        elif policy == "mix":
            deterministic = sampler(
                samples=1,
                temperature=base_temperature,
                stochastic=False,
            )
            stochastic = sampler(
                samples=total_samples - 1,
                temperature=base_temperature,
                stochastic=True,
            )
            chunks.extend(chunk for chunk in (deterministic, stochastic) if chunk is not None)
        elif policy == "ens":
            counts = _split_counts(total_samples, ensemble_weights)
            for count, temperature in zip(counts, ensemble_temperatures):
                chunk = sampler(
                    samples=count,
                    temperature=temperature,
                    stochastic=True,
                )
                if chunk is not None:
                    chunks.append(chunk)
        elif policy == "mixens":
            deterministic = sampler(
                samples=1,
                temperature=base_temperature,
                stochastic=False,
            )
            if deterministic is not None:
                chunks.append(deterministic)
            counts = _split_counts(total_samples - 1, ensemble_weights)
            for count, temperature in zip(counts, ensemble_temperatures):
                chunk = sampler(
                    samples=count,
                    temperature=temperature,
                    stochastic=True,
                )
                if chunk is not None:
                    chunks.append(chunk)
        else:
            raise ValueError(f"Unsupported decoding policy: {policy!r}")
        return _combine_sampled(chunks)

    if family == "direct":
        samples = int(spec["samples"])
        return compose(
            total_samples=samples,
            sampler=lambda *, samples, temperature, stochastic: _direct_chunk(
                direct_solver,
                instance,
                samples=samples,
                temperature=temperature,
                stochastic=stochastic,
                sample_batch_size=sample_batch_size,
                device=device,
                generator=generator,
            ),
        )
    if family == "masked":
        samples = int(spec["samples"])
        return compose(
            total_samples=samples,
            sampler=lambda *, samples, temperature, stochastic: _masked_chunk(
                masked_solver,
                instance,
                samples=samples,
                temperature=temperature,
                stochastic=stochastic,
                sample_batch_size=sample_batch_size,
                device=device,
                generator=generator,
            ),
        )
    if family == "sequential":
        budget = int(spec["budget"])
        samples = max(1, budget // max(1, int(instance.num_services)))
        return compose(
            total_samples=samples,
            sampler=lambda *, samples, temperature, stochastic: _sequential_chunk(
                sequential_solver,
                instance,
                samples=samples,
                temperature=temperature,
                stochastic=stochastic,
                sample_batch_size=sample_batch_size,
                device=device,
                generator=generator,
            ),
        )
    raise ValueError(f"Cannot sample method family {family!r}")


def _payload(result: Any, pool_best: float, sampled: Mapping[str, Any] | None) -> dict[str, Any]:
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
        "raw_any_feasible": bool(metrics.get("raw_any_feasible", result.success)),
        "raw_feasible_count": int(metrics.get("raw_feasible_count", int(result.success))),
        "num_raw_proposals": int(metrics.get("num_raw_proposals", 1)),
        "raw_feasible_rate": metrics.get("raw_feasible_rate"),
        "raw_unique_rate": metrics.get("raw_unique_rate"),
        "raw_pairwise_hamming": metrics.get("raw_pairwise_hamming"),
        "fallback_invoked": bool(metrics.get("fallback_invoked", False)),
        "fallback_success": bool(metrics.get("fallback_success", False)),
        "sampling_seconds": float(metrics.get("sampling_seconds", 0.0)),
        "verification_seconds": float(metrics.get("verification_seconds", 0.0)),
        "fallback_seconds": float(metrics.get("fallback_seconds", 0.0)),
        "total_seconds": float(metrics.get("total_seconds", 0.0)),
        "model_forwards": None if sampled is None else sampled.get("model_forwards"),
        "completed_rate": None if sampled is None else sampled.get("completed_rate"),
        "realized_budget": None if sampled is None else sampled.get("realized_budget"),
    }


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


def _run_setting_seed(
    root: Path,
    *,
    setting: str,
    spec: Mapping[str, Any],
    seed: int,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    training: Mapping[str, Any],
    training_freeze: Path,
    sequential_checkpoint_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    dataset_root = _resolve(root, spec["dataset_root"])
    if not dataset_root.is_dir():
        if protocol["skip_missing_datasets"]:
            return {
                "setting": setting,
                "seed": int(seed),
                "records_completed": 0,
                "expected": 0,
                "skipped_missing_dataset": _relative(root, dataset_root.parent)
                + f"/{dataset_root.name}",
            }
        raise FileNotFoundError(f"Missing dataset root for {setting}: {dataset_root}")

    freeze = audit_dataset_freeze(dataset_root)
    dataset_hash = file_sha256(dataset_root / "dataset_freeze.json")
    training_hash = file_sha256(training_freeze)
    dataset = LabeledDeploymentDataset(
        dataset_root,
        partitions=tuple(spec["partitions"]),
        require_freeze=True,
    )
    indices = _limited_indices(dataset, protocol["max_instances_per_partition"])
    device = _device(str(protocol["device"]))
    direct = load_stage3_solver(
        _resolve(root, training["runs"][str(seed)]["direct"]["paths"]["best.pt"]),
        dataset,
        device,
    )
    masked = load_stage3_solver(
        _resolve(root, training["runs"][str(seed)]["masked_conditional"]["paths"]["best.pt"]),
        dataset,
        device,
    )
    if masked.schedule is None:
        raise ValueError("Masked checkpoint has no absorbing-MASK schedule.")
    sequential_checkpoint = _SEQUENTIAL._sequential_checkpoint(
        root,
        sequential_checkpoint_root,
        int(seed),
    )
    sequential_hash = file_sha256(sequential_checkpoint)
    sequential = _SEQUENTIAL.load_sequential_solver(sequential_checkpoint, dataset, device)

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
            if method["family"] == "greedy":
                result = solve_greedy_local(item.instance)
                method_results[method_id] = _payload(result, pool_best, None)
                continue

            sampled = _sample_method(
                method_id=method_id,
                spec=method,
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
                    int(sampled["proposals"].shape[0]),
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
                "seed": int(seed),
                "instance_id": item.instance.instance_id,
                "partition": item.partition,
                "num_services": int(item.instance.num_services),
                "num_devices": int(item.instance.num_devices),
                "num_dependencies": int(item.instance.dependency_index.shape[1]),
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
        "seed": int(seed),
        "records_completed": completed,
        "expected": len(indices),
    }


def run_evaluation(
    root: Path,
    *,
    settings: tuple[str, ...],
    selected_seeds: tuple[int, ...],
    methods: tuple[str, ...],
    training_freeze: Path,
    sequential_checkpoint_root: Path,
    output_root: Path,
    device_name: str,
    max_instances_per_partition: int | None,
    fallback_max_search_nodes: int,
    skip_missing_datasets: bool,
) -> dict[str, Any]:
    protocol = _protocol(
        root=root,
        settings=settings,
        selected_seeds=selected_seeds,
        methods=methods,
        training_freeze=training_freeze,
        sequential_checkpoint_root=sequential_checkpoint_root,
        output_root=output_root,
        device=device_name,
        max_instances_per_partition=max_instances_per_partition,
        fallback_max_search_nodes=fallback_max_search_nodes,
        skip_missing_datasets=skip_missing_datasets,
    )
    protocol_hash = _SEQUENTIAL._hash_payload(protocol)
    seed_everything(int(protocol["evaluation_seed"]), deterministic=True)
    training = _read_json(training_freeze)
    if training.get("scope") != "phase6e_e_stage39_forward_budget_training_freeze":
        raise ValueError("Expected Stage 3.9 ten-seed training freeze.")
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
                    seed=int(seed),
                    protocol=protocol,
                    protocol_hash=protocol_hash,
                    training=training,
                    training_freeze=training_freeze,
                    sequential_checkpoint_root=sequential_checkpoint_root,
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
    total_proposals = sum(int(payload["num_raw_proposals"]) for payload in payloads)
    total_feasible = sum(int(payload["raw_feasible_count"]) for payload in payloads)
    source_counts: dict[str, int] = defaultdict(int)
    for payload in payloads:
        source_counts[str(payload["source"])] += 1
    return {
        "records": records,
        "success_rate": mean(float(payload["success"]) for payload in payloads),
        "mean_gap": _mean([payload["gap_to_pool_best"] for payload in payloads]),
        "gap_std": _std([payload["gap_to_pool_best"] for payload in payloads]),
        "raw_any_feasible_rate": _mean(
            [float(payload["raw_any_feasible"]) for payload in payloads]
        ),
        "proposal_feasible_rate": (
            total_feasible / total_proposals if total_proposals else None
        ),
        "mean_raw_gap": _mean([payload["raw_gap_to_pool_best"] for payload in payloads]),
        "fallback_rate": _mean(
            [float(payload["fallback_invoked"]) for payload in payloads]
        ),
        "mean_total_seconds": _mean([payload["total_seconds"] for payload in payloads]),
        "mean_model_forwards": _mean([payload["model_forwards"] for payload in payloads]),
        "mean_completed_rate": _mean([payload["completed_rate"] for payload in payloads]),
        "mean_realized_budget": _mean([payload["realized_budget"] for payload in payloads]),
        "source_rates": {
            source: count / records if records else None
            for source, count in sorted(source_counts.items())
        },
    }


def finalize(root: Path, *, output_root: Path) -> dict[str, Any]:
    rows = _records(output_root)
    if not rows:
        raise RuntimeError(f"No records found under {output_root / 'records'}")
    protocol = _read_json(output_root / "protocol.json")
    methods = tuple(protocol["methods"].keys())
    by_setting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setting[str(row["setting"])].append(row)

    evidence: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "methods": protocol["methods"],
        "overall": {},
        "by_setting": {},
    }
    for method in methods:
        evidence["overall"][method] = _aggregate(
            [row["methods"][method] for row in rows]
        )
    for setting, setting_rows in sorted(by_setting.items()):
        evidence["by_setting"][setting] = {
            method: _aggregate([row["methods"][method] for row in setting_rows])
            for method in methods
        }

    evidence_path = output_root / "decoding_enhancement_probe_evidence.json"
    report_path = output_root / "decoding_enhancement_probe_report.md"
    write_json(evidence_path, evidence)
    report_path.write_text(_report(evidence), encoding="utf-8")
    return {
        "scope": EVIDENCE_SCOPE,
        "records": len(rows),
        "evidence": _relative(root, evidence_path),
        "report": _relative(root, report_path),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _method_row(method: str, row: Mapping[str, Any]) -> str:
    return (
        f"| {method} | {row['records']} | {_pct(row['success_rate'])} | "
        f"{_pct(row['raw_any_feasible_rate'])} | {_pct(row['proposal_feasible_rate'])} | "
        f"{_pct(row['mean_gap'])} | {_pct(row['mean_raw_gap'])} | "
        f"{_pct(row['fallback_rate'])} | {_num(row['mean_total_seconds'])} |"
    )


def _report(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Neural Decoding Enhancement Probe",
        "",
        "This probe excludes the latency-aware heuristic. Greedy is a single-pass",
        "topological reference. Learned methods use hard verification, exact-latency",
        "selection among verified proposals, and fallback only when no verified",
        "learned proposal exists.",
        "",
        f"Records: {evidence['records']}",
        "",
        "## Overall",
        "",
        "| Method | Records | Success | Any feasible | Proposal feasible | Gap | Raw gap | Fallback | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in evidence["overall"].items():
        lines.append(_method_row(method, row))
    lines.extend(["", "## By Setting", ""])
    for setting, methods in evidence["by_setting"].items():
        lines.extend(
            [
                f"### {setting}",
                "",
                "| Method | Records | Success | Any feasible | Proposal feasible | Gap | Raw gap | Fallback | Time (s) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in methods.items():
            lines.append(_method_row(method, row))
        lines.append("")
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "- Treat this as a decoding probe, not frozen paper evidence.",
            "- If a larger masked budget beats Greedy only by much higher runtime, keep it as a sensitivity result.",
            "- If Direct or Sequential improves mainly by larger K, compare under the same B_NN budget before making claims.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "finalize", "all"))
    parser.add_argument("--settings", default="sealed_id,controlled_shift,realistic_profile")
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
        default="artifacts/phase6f-decoding-enhancement-probe",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-instances-per-partition", type=int, default=4)
    parser.add_argument("--fallback-max-search-nodes", type=int, default=30000)
    parser.add_argument("--no-skip-missing-datasets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = _resolve(root, args.output_root)
    settings = _parse_csv(args.settings, default=DATASET_SPECS)
    unknown_settings = sorted(set(settings) - set(DATASET_SPECS))
    if unknown_settings:
        raise ValueError(f"Unsupported settings: {unknown_settings}")
    methods = _parse_csv(args.methods, default=DEFAULT_METHODS)
    if args.max_instances_per_partition is not None and args.max_instances_per_partition < 1:
        raise ValueError("--max-instances-per-partition must be positive.")
    kwargs = {
        "settings": settings,
        "selected_seeds": _parse_seeds(args.seeds),
        "methods": methods,
        "training_freeze": _resolve(root, args.training_freeze),
        "sequential_checkpoint_root": _resolve(root, args.sequential_checkpoint_root),
        "output_root": output_root,
        "device_name": args.device,
        "max_instances_per_partition": args.max_instances_per_partition,
        "fallback_max_search_nodes": int(args.fallback_max_search_nodes),
        "skip_missing_datasets": not bool(args.no_skip_missing_datasets),
    }
    if args.action in {"run", "all"}:
        print(run_evaluation(root, **kwargs), flush=True)
    if args.action in {"finalize", "all"}:
        print(finalize(root, output_root=output_root), flush=True)


if __name__ == "__main__":
    main()
