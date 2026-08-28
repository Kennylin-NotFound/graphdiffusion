"""Validate and freeze the completed Stage 3 one-seed training runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from gdm_factor_diffusion.common.config import load_config
from gdm_factor_diffusion.common.logging import write_json


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _rank(record: dict[str, Any]) -> tuple[float, float, float, float]:
    gap = record["mean_pre_fallback_gap"]
    return (
        -float(record["pre_fallback_success_rate"]),
        float("inf") if gap is None else float(gap),
        -float(record["raw_any_feasibility"]),
        float(record["mean_online_seconds"]),
    )


def _summary(run_directory: Path, expected_config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        name: run_directory / name
        for name in ("config.json", "metrics.jsonl", "best.pt", "latest.pt")
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_directory.name}: {missing}")
    if _read_json(paths["config.json"]) != expected_config:
        raise ValueError(f"Config drift in {run_directory.name}.")

    records = _read_jsonl(paths["metrics.jsonl"])
    train = [record for record in records if record.get("type") == "train"]
    selection = [
        record for record in records if record.get("type") == "checkpoint_selection"
    ]
    if not train or not selection:
        raise ValueError(f"Missing training or selection records in {run_directory.name}.")
    final_step = int(train[-1]["step"])
    if final_step != int(expected_config["optimization"]["max_steps"]):
        raise ValueError(f"Run {run_directory.name} stopped at step {final_step}.")
    if len(selection) != final_step // int(
        expected_config["optimization"]["validation_interval"]
    ):
        raise ValueError(f"Unexpected selection count in {run_directory.name}.")

    best_record = min(selection, key=_rank)
    best_payload = torch.load(paths["best.pt"], map_location="cpu", weights_only=False)
    latest_payload = torch.load(
        paths["latest.pt"], map_location="cpu", weights_only=False
    )
    if int(best_payload["step"]) != int(best_record["step"]):
        raise ValueError(f"Best-checkpoint mismatch in {run_directory.name}.")
    if int(latest_payload["step"]) != final_step:
        raise ValueError(f"Latest-checkpoint mismatch in {run_directory.name}.")

    return {
        "run": run_directory.name,
        "final_step": final_step,
        "train_record_count": len(train),
        "selection_record_count": len(selection),
        "best_step": int(best_record["step"]),
        "best_selection": {
            key: best_record[key]
            for key in (
                "final_verified_rate",
                "pre_fallback_success_rate",
                "mean_pre_fallback_gap",
                "raw_any_feasibility",
                "mean_online_seconds",
            )
        },
        "final_selection": {
            key: selection[-1][key]
            for key in (
                "final_verified_rate",
                "pre_fallback_success_rate",
                "mean_pre_fallback_gap",
                "raw_any_feasibility",
                "mean_online_seconds",
            )
        },
        "sha256": {name: _sha256(path) for name, path in paths.items()},
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "training_phase6e_e_stage3_pilot.yaml"
    expected_config = load_config(config_path)
    training_root = root / "artifacts" / "phase6e-e-stage3-training"
    pilot_root = root / "artifacts" / "phase6e-e-stage3-pilot"
    if pilot_root.exists():
        raise ValueError("The one-time Stage 3 pilot has already been opened.")

    runs = {
        model_kind: _summary(
            training_root / f"{model_kind}-seed2026070111", expected_config
        )
        for model_kind in ("direct", "masked_conditional")
    }
    direct = runs["direct"]["best_selection"]
    masked = runs["masked_conditional"]["best_selection"]
    comparison = {
        "relative_gap_improvement": (
            float(direct["mean_pre_fallback_gap"])
            - float(masked["mean_pre_fallback_gap"])
        )
        / float(direct["mean_pre_fallback_gap"]),
        "raw_feasibility_percentage_point_improvement": 100.0
        * (
            float(masked["raw_any_feasibility"])
            - float(direct["raw_any_feasibility"])
        ),
        "online_time_ratio": float(masked["mean_online_seconds"])
        / float(direct["mean_online_seconds"]),
        "interpretation": "checkpoint_selection_only_not_pilot_evidence",
    }
    pretraining_freeze = (
        root / "artifacts" / "phase6e-e-stage3" / "pretraining_freeze.json"
    )
    freeze = {
        "schema_version": "1.0",
        "phase": "6E-E Stage 3 one-seed training freeze",
        "formal_training_complete": True,
        "pilot_opened": False,
        "regression_test_count": 117,
        "config_sha256": _sha256(config_path),
        "pretraining_freeze_sha256": _sha256(pretraining_freeze),
        "runs": runs,
        "checkpoint_selection_comparison": comparison,
    }
    destination = training_root / "training_freeze.json"
    if destination.exists() and _read_json(destination) != freeze:
        raise ValueError("Existing Stage 3 training freeze disagrees with current runs.")
    write_json(destination, freeze)
    print(
        "stage3_training_complete=True pilot_opened=False "
        f"direct_best={runs['direct']['best_step']} "
        f"masked_best={runs['masked_conditional']['best_step']}"
    )
    print(f"freeze={destination} sha256={_sha256(destination)}")


if __name__ == "__main__":
    main()
