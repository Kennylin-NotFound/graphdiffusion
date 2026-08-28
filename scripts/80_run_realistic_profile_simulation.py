"""Realistic-profile simulation evaluation for the absorbing-MASK solver.

This is a thin wrapper around the controlled-shift evaluation implementation.
It reuses the same verified inference and aggregation code while assigning a
separate scope and default partitions for realistic-profile evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path
from typing import Any


DEFAULT_PARTITIONS = ("profile_id", "profile_branched", "profile_high_sharing")


def _load_controlled_module() -> Any:
    path = Path(__file__).resolve().with_name("78_run_phase6e_e_controlled_shift_evaluation.py")
    spec = importlib.util.spec_from_file_location("_controlled_shift_eval_for_realistic", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluation module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCOPE = "phase6e_e_realistic_profile"
    module.RECORD_SCOPE = f"{module.SCOPE}_seed_instance_record"
    module.EVIDENCE_SCOPE = f"{module.SCOPE}_evidence"
    module.DEFAULT_PARTITIONS = DEFAULT_PARTITIONS
    return module


def _parse_csv(module: Any, value: str | None) -> tuple[str, ...]:
    return module._parse_csv(value, default=DEFAULT_PARTITIONS)


def _copy_realistic_outputs(output_root: Path, result: dict[str, Any]) -> dict[str, Any]:
    evidence_src = output_root / "controlled_shift_evidence.json"
    report_src = output_root / "controlled_shift_report.md"
    evidence_dst = output_root / "realistic_profile_evidence.json"
    report_dst = output_root / "realistic_profile_report.md"
    copied = dict(result)
    if evidence_src.is_file():
        shutil.copy2(evidence_src, evidence_dst)
        copied["evidence"] = str(evidence_dst)
    if report_src.is_file():
        shutil.copy2(report_src, report_dst)
        copied["report"] = str(report_dst)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "finalize", "all"))
    parser.add_argument(
        "--dataset-root",
        default="artifacts/datasets/phase6e-e-realistic-profile",
    )
    parser.add_argument(
        "--training-freeze",
        default=(
            "artifacts/phase6e-e-stage39-10seed-training/"
            "ten_seed_training_freeze.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase6e-e-realistic-profile-evaluation-10seed",
    )
    parser.add_argument("--partitions", default=",".join(DEFAULT_PARTITIONS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in range(2026070111, 2026070121)))
    parser.add_argument("--max-instances-per-partition", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--method-profile",
        choices=("full", "core", "lean"),
        default="core",
    )
    parser.add_argument("--repair-candidate-limit", type=int, default=16)
    parser.add_argument("--repair-max-moves", type=int, default=8)
    parser.add_argument("--fallback-max-search-nodes", type=int, default=30000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = _load_controlled_module()
    root = Path(__file__).resolve().parents[1]
    partitions = _parse_csv(module, args.partitions)
    selected_seeds = module._parse_seed_list(args.seeds)
    if args.max_instances_per_partition is not None and args.max_instances_per_partition < 1:
        raise ValueError("--max-instances-per-partition must be positive.")
    kwargs = {
        "dataset_root": module._resolve(root, args.dataset_root),
        "training_freeze": module._resolve(root, args.training_freeze),
        "output_root": module._resolve(root, args.output_root),
        "partitions": partitions,
        "max_instances_per_partition": args.max_instances_per_partition,
        "device_name": args.device,
        "selected_seeds": selected_seeds,
        "method_profile": module._validate_method_profile(args.method_profile),
        "repair_candidate_limit": args.repair_candidate_limit,
        "repair_max_moves": args.repair_max_moves,
        "fallback_max_search_nodes": args.fallback_max_search_nodes,
    }
    if args.action in {"run", "all"}:
        print(module.run_evaluation(root, **kwargs), flush=True)
    if args.action in {"finalize", "all"}:
        result = module.finalize(root, **kwargs)
        print(_copy_realistic_outputs(kwargs["output_root"], result), flush=True)


if __name__ == "__main__":
    main()
