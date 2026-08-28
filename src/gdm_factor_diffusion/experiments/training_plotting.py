"""Diagnostic training-curve export from append-only production metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gdm_factor_diffusion.common.logging import write_json


def export_training_curves(
    run_directory: str | Path,
    *,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Export one-run diagnostics without treating them as multi-seed evidence."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run_directory)
    records = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [record for record in records if record["split"] == "train"]
    denoising = [
        record for record in records if record["split"] == "validation_denoising"
    ]
    constrained = [
        record for record in records if record["split"] == "validation_constrained"
    ]
    if not train or not denoising or not constrained:
        raise ValueError("Training curve export requires train and validation metrics.")

    output = Path(output_directory) if output_directory else run / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8})
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))

    axes[0, 0].plot(
        [record["step"] for record in train],
        [record["loss_total"] for record in train],
        label="train",
    )
    axes[0, 0].plot(
        [record["step"] for record in denoising],
        [record["loss_total"] for record in denoising],
        label="validation",
    )
    axes[0, 0].set_ylabel("Total loss")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        [record["step"] for record in denoising],
        [100.0 * record["clean_accuracy"] for record in denoising],
    )
    axes[0, 1].set_ylabel("Validation clean accuracy (%)")

    axes[1, 0].plot(
        [record["step"] for record in constrained],
        [100.0 * record["mean_gap_to_pool_best"] for record in constrained],
    )
    axes[1, 0].set_ylabel("Verified gap to pool best (%)")

    axes[1, 1].plot(
        [record["step"] for record in constrained],
        [100.0 * record["mean_raw_feasible_rate"] for record in constrained],
        label="raw feasible rate",
    )
    axes[1, 1].plot(
        [record["step"] for record in constrained],
        [
            100.0
            * record["learned_wins_over_fallback"]
            / record["instances"]
            for record in constrained
        ],
        label="wins over fallback",
    )
    axes[1, 1].set_ylabel("Validation rate (%)")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xlabel("Training step")
        axis.grid(True, linestyle=":", linewidth=0.5)
    paths = []
    for suffix in (".png", ".pdf"):
        path = output / f"training_diagnostics{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=300)
        paths.append(str(path.resolve()))
    plt.close(figure)
    payload = {
        "schema_version": "1.0",
        "run_directory": str(run.resolve()),
        "scope": "single_seed_diagnostic",
        "figures": {"training_diagnostics": paths},
    }
    write_json(output / "training_figure_manifest.json", payload)
    return payload
