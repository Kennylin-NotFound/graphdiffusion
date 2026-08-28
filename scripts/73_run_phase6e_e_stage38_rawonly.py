"""Run a raw-only Stage 3.8 evaluation without repair or fallback.

This script reuses the frozen Stage 3.8 checkpoints and sealed dataset, but
sets the inference post-processing switches to raw proposal verification only.
It is intended as a diagnostic evidence table, not as a replacement for the
frozen Stage 3.8 decision lock.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping

import torch

from gdm_factor_diffusion.common.logging import write_json
from gdm_factor_diffusion.common.seed import derive_seed, seed_everything
from gdm_factor_diffusion.inference import (
    InferenceConfig,
    MaskedDecodeConfig,
    solve_with_direct_predictor,
    solve_with_masked_model,
)
from gdm_factor_diffusion.training import LabeledDeploymentDataset
from gdm_factor_diffusion.experiments.phase6ee_stage3 import load_stage3_solver
from gdm_factor_diffusion.experiments.phase6ee_stage38 import (
    _load_confirmed_inputs,
    _read_json,
    _resolve,
    verify_stage38_lock,
)
from gdm_factor_diffusion.experiments.schema import file_sha256


METHODS = ("direct_k64", "masked_deterministic_k1", "masked_diffusion_k8")
METHOD_LABELS = {
    "direct_k64": "Direct K=64",
    "masked_deterministic_k1": "Masked Det. K=1",
    "masked_diffusion_k8": "Masked Diff. K=8",
}
RAW_RECORD_SCOPE = "phase6e_e_stage38_rawonly_seed_instance_record"
RAW_EVIDENCE_SCOPE = "phase6e_e_stage38_rawonly_evidence"
SAMPLING_SEED_NAMESPACE = "stage38_original"


def _raw_inference_config(lock: Mapping[str, Any], samples: int) -> InferenceConfig:
    post = lock["postprocessing"]
    return InferenceConfig(
        num_samples=samples,
        sample_batch_size=min(int(lock["methods"]["sample_batch_size"]), samples),
        repair_max_moves=int(post["repair_max_moves"]),
        fallback_max_search_nodes=int(post["fallback_max_search_nodes"]),
        enable_repair=False,
        enable_fallback=False,
        always_include_fallback=False,
    )


def _payload(result: Any, pool_best: float) -> dict[str, Any]:
    metrics = result.metrics
    best_raw = metrics.get("best_raw_objective")
    return {
        "raw_success": best_raw is not None,
        "raw_objective": best_raw,
        "raw_gap_to_pool_best": None
        if best_raw is None
        else float(best_raw) / pool_best - 1.0,
        "raw_any_feasible": bool(metrics["raw_any_feasible"]),
        "raw_feasible_count": int(metrics["raw_feasible_count"]),
        "num_raw_proposals": int(metrics["num_raw_proposals"]),
        "raw_feasible_rate": float(metrics["raw_feasible_rate"]),
        "raw_capacity_violation_rate": float(metrics["raw_capacity_violation_rate"]),
        "raw_link_violation_rate": float(metrics["raw_link_violation_rate"]),
        "source": result.source,
        "success": bool(result.success),
        "objective": result.objective,
        "gap_to_pool_best": None
        if result.objective is None
        else float(result.objective) / pool_best - 1.0,
        "sampling_seconds": float(metrics["sampling_seconds"]),
        "verification_seconds": float(metrics["verification_seconds"]),
        "total_seconds": float(metrics["total_seconds"]),
    }


def _aggregate(records: list[Mapping[str, Any]], method_id: str) -> dict[str, Any]:
    rows = [record["methods"][method_id] for record in records]
    raw_gaps = [
        float(row["raw_gap_to_pool_best"])
        for row in rows
        if row["raw_success"] and row["raw_gap_to_pool_best"] is not None
    ]
    return {
        "records": len(rows),
        "raw_success_rate": mean(float(row["raw_success"]) for row in rows),
        "raw_any_feasibility": mean(float(row["raw_any_feasible"]) for row in rows),
        "mean_raw_feasible_proposal_rate": mean(float(row["raw_feasible_rate"]) for row in rows),
        "mean_raw_gap_to_pool_best": mean(raw_gaps) if raw_gaps else None,
        "raw_gap_std": pstdev(raw_gaps) if len(raw_gaps) > 1 else 0.0,
        "mean_capacity_violation_rate": mean(float(row["raw_capacity_violation_rate"]) for row in rows),
        "mean_link_violation_rate": mean(float(row["raw_link_violation_rate"]) for row in rows),
        "mean_sampling_seconds": mean(float(row["sampling_seconds"]) for row in rows),
        "mean_total_seconds": mean(float(row["total_seconds"]) for row in rows),
    }


def _sign_test_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, idx) for idx in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _write_report(path: Path, evidence: Mapping[str, Any]) -> None:
    overall = evidence["overall"]
    per_seed = evidence["per_seed"]
    lines = [
        "# Phase 6E-E Stage 3.8 Raw-Only Evaluation",
        "",
        "This diagnostic reuses the frozen Stage 3.8 checkpoints and sealed",
        "dataset, but disables both bounded repair and deterministic fallback.",
        "Therefore the reported gap is the best feasible raw-proposal gap only.",
        "",
        "## Overall",
        "",
        "| Method | Raw success | Raw feasible proposal rate | Raw gap | Capacity violation | Link violation | Total time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = overall[method]
        raw_gap = value["mean_raw_gap_to_pool_best"]
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{100 * value['raw_success_rate']:.2f}% | "
            f"{100 * value['mean_raw_feasible_proposal_rate']:.2f}% | "
            f"{'N/A' if raw_gap is None else f'{100 * raw_gap:.3f}%'} | "
            f"{100 * value['mean_capacity_violation_rate']:.2f}% | "
            f"{100 * value['mean_link_violation_rate']:.2f}% | "
            f"{value['mean_total_seconds']:.3f} s |"
        )
    lines.extend([
        "",
        "## Seed-Level",
        "",
        "| Seed | Method | Raw success | Raw feasible proposal rate | Raw gap | Total time |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for seed in sorted(per_seed):
        for method in METHODS:
            value = per_seed[seed][method]
            raw_gap = value["mean_raw_gap_to_pool_best"]
            lines.append(
                f"| {seed} | {METHOD_LABELS[method]} | "
                f"{100 * value['raw_success_rate']:.2f}% | "
                f"{100 * value['mean_raw_feasible_proposal_rate']:.2f}% | "
                f"{'N/A' if raw_gap is None else f'{100 * raw_gap:.3f}%'} | "
                f"{value['mean_total_seconds']:.3f} s |"
            )
    gate = evidence["direct_vs_masked_diffusion"]
    lines.extend([
        "",
        "## Direct vs. Masked Diffusion",
        "",
        f"- Raw instance wins/losses/ties: {gate['wins']} / {gate['losses']} / {gate['ties']}.",
        f"- Raw paired sign-test p-value: {gate['sign_test_pvalue']:.6f}.",
        f"- Raw gap relative improvement: {100 * gate['relative_raw_gap_improvement']:.2f}%.",
        f"- Raw success-rate improvement: {100 * gate['raw_success_rate_improvement_pp']:.2f} percentage points.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rawonly() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "phase6e_e_stage38_sealed.yaml"
    lock_path = root / "artifacts" / "phase6e-e-stage38" / "preparation_lock.json"
    lock = verify_stage38_lock(lock_path, implementation_root=root)
    device = torch.device(lock["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Stage 3.8 contract.")
    seed_everything(int(lock["evaluation_seed"]), deterministic=bool(lock["deterministic"]))
    dataset_root, dataset_hash, training_path, training = _load_confirmed_inputs(root, lock)
    dataset = LabeledDeploymentDataset(
        dataset_root, partitions=(lock["sealed_partition"],), require_freeze=True
    )
    output_root = root / "artifacts" / "phase6e-e-stage38-rawonly-evaluation"
    records_root = output_root / "records"
    completed = 0
    methods = lock["methods"]
    for training_seed in lock["training_seeds"]:
        seed_key = str(training_seed)
        direct_entry = training["runs"][seed_key]["direct"]
        masked_entry = training["runs"][seed_key]["masked_conditional"]
        direct = load_stage3_solver(_resolve(root, direct_entry["paths"]["best.pt"]), dataset, device)
        masked = load_stage3_solver(_resolve(root, masked_entry["paths"]["best.pt"]), dataset, device)
        if masked.schedule is None:
            raise ValueError("Stage 3.8 masked checkpoint has no schedule.")

        warm = dataset[0]
        solve_with_direct_predictor(
            direct.model,
            warm.instance,
            direct.feature_schema,
            config=_raw_inference_config(lock, int(methods["direct_samples"])),
            device=device,
            generator=torch.Generator(device=device).manual_seed(
                derive_seed(int(lock["evaluation_seed"]), f"warm:direct:{training_seed}")
            ),
        )
        for samples, stochastic in ((1, False), (int(methods["diffusion_samples"]), True)):
            solve_with_masked_model(
                masked.model,
                warm.instance,
                masked.schedule,
                masked.feature_schema,
                decode_config=MaskedDecodeConfig(
                    num_samples=samples,
                    sample_batch_size=min(int(methods["sample_batch_size"]), samples),
                    stochastic=stochastic,
                    temperature=float(methods["temperature"]),
                ),
                inference_config=_raw_inference_config(lock, samples),
                device=device,
                generator=torch.Generator(device=device).manual_seed(
                    derive_seed(int(lock["evaluation_seed"]), f"warm:masked:{training_seed}:{samples}")
                ),
            )

        for index in range(len(dataset)):
            item = dataset[index]
            record_path = records_root / seed_key / f"{item.instance.instance_id}.json"
            if record_path.exists():
                row = _read_json(record_path)
                if (
                    row.get("scope") == RAW_RECORD_SCOPE
                    and row.get("sampling_seed_namespace") == SAMPLING_SEED_NAMESPACE
                    and row.get("preparation_lock_sha256") == file_sha256(lock_path)
                    and row.get("dataset_freeze_sha256") == dataset_hash
                    and row.get("training_freeze_sha256") == file_sha256(training_path)
                ):
                    completed += 1
                    continue
                record_path.unlink()

            pool_best = float(min(item.pool.latencies))
            direct_result = solve_with_direct_predictor(
                direct.model,
                item.instance,
                direct.feature_schema,
                config=_raw_inference_config(lock, int(methods["direct_samples"])),
                device=device,
                generator=torch.Generator(device=device).manual_seed(
                    derive_seed(int(lock["evaluation_seed"]), f"direct:{training_seed}:{item.instance.instance_id}")
                ),
            )
            results = {"direct_k64": _payload(direct_result, pool_best)}
            for method_id, samples, stochastic in (
                ("masked_deterministic_k1", 1, False),
                ("masked_diffusion_k8", int(methods["diffusion_samples"]), True),
            ):
                result = solve_with_masked_model(
                    masked.model,
                    item.instance,
                    masked.schedule,
                    masked.feature_schema,
                    decode_config=MaskedDecodeConfig(
                        num_samples=samples,
                        sample_batch_size=min(int(methods["sample_batch_size"]), samples),
                        stochastic=stochastic,
                        temperature=float(methods["temperature"]),
                    ),
                    inference_config=_raw_inference_config(lock, samples),
                    device=device,
                    generator=torch.Generator(device=device).manual_seed(
                        derive_seed(int(lock["evaluation_seed"]), f"{method_id}:{training_seed}:{item.instance.instance_id}")
                    ),
                )
                results[method_id] = _payload(result, pool_best)

            write_json(record_path, {
                "schema_version": "1.0",
                "scope": RAW_RECORD_SCOPE,
                "training_seed": int(training_seed),
                "instance_id": item.instance.instance_id,
                "pool_best": pool_best,
                "stage38_config_sha256": file_sha256(config),
                "preparation_lock_sha256": file_sha256(lock_path),
                "dataset_freeze_sha256": dataset_hash,
                "training_freeze_sha256": file_sha256(training_path),
                "sampling_seed_namespace": SAMPLING_SEED_NAMESPACE,
                "postprocessing": {
                    "enable_repair": False,
                    "enable_fallback": False,
                    "always_include_fallback": False,
                },
                "methods": results,
            })
            completed += 1

    records = [
        _read_json(path)
        for path in sorted(records_root.glob("*/*.json"))
        if _read_json(path).get("scope") == RAW_RECORD_SCOPE
    ]
    per_seed: dict[str, Any] = {}
    for training_seed in lock["training_seeds"]:
        seed_key = str(training_seed)
        seed_records = [row for row in records if int(row["training_seed"]) == int(training_seed)]
        per_seed[seed_key] = {method: _aggregate(seed_records, method) for method in METHODS}
    overall = {method: _aggregate(records, method) for method in METHODS}

    by_instance: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_instance[str(row["instance_id"])].append(row)
    wins = losses = ties = 0
    for rows in by_instance.values():
        scores = {}
        for method in ("direct_k64", "masked_diffusion_k8"):
            values = [row["methods"][method] for row in rows]
            success = mean(float(value["raw_success"]) for value in values)
            gaps = [
                float(value["raw_gap_to_pool_best"])
                for value in values
                if value["raw_success"] and value["raw_gap_to_pool_best"] is not None
            ]
            scores[method] = (-success, mean(gaps) if gaps else float("inf"))
        if scores["masked_diffusion_k8"] < scores["direct_k64"]:
            wins += 1
        elif scores["direct_k64"] < scores["masked_diffusion_k8"]:
            losses += 1
        else:
            ties += 1

    direct = overall["direct_k64"]
    masked = overall["masked_diffusion_k8"]
    evidence = {
        "schema_version": "1.0",
        "scope": RAW_EVIDENCE_SCOPE,
        "records": len(records),
        "expected": len(dataset) * len(lock["training_seeds"]),
        "stage38_config_sha256": file_sha256(config),
        "preparation_lock_sha256": file_sha256(lock_path),
        "dataset_freeze_sha256": dataset_hash,
        "training_freeze_sha256": file_sha256(training_path),
        "sampling_seed_namespace": SAMPLING_SEED_NAMESPACE,
        "overall": overall,
        "per_seed": per_seed,
        "direct_vs_masked_diffusion": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "sign_test_pvalue": _sign_test_pvalue(wins, losses),
            "relative_raw_gap_improvement": (
                (direct["mean_raw_gap_to_pool_best"] - masked["mean_raw_gap_to_pool_best"])
                / direct["mean_raw_gap_to_pool_best"]
                if direct["mean_raw_gap_to_pool_best"] is not None and masked["mean_raw_gap_to_pool_best"] is not None
                else None
            ),
            "raw_success_rate_improvement_pp": masked["raw_success_rate"] - direct["raw_success_rate"],
            "raw_feasible_proposal_rate_improvement_pp": (
                masked["mean_raw_feasible_proposal_rate"]
                - direct["mean_raw_feasible_proposal_rate"]
            ),
        },
    }
    write_json(output_root / "rawonly_evidence.json", evidence)
    _write_report(output_root / "rawonly_report.md", evidence)
    return {
        "records_completed": completed,
        "expected": evidence["expected"],
        "evidence": str(output_root / "rawonly_evidence.json"),
        "report": str(output_root / "rawonly_report.md"),
    }


if __name__ == "__main__":
    print(run_rawonly())
