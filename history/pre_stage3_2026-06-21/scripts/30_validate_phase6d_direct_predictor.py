"""Validate the Phase 6D-B direct predictor with a bounded overfit gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gdm_factor_diffusion.diffusion import CategoricalSchedule
from gdm_factor_diffusion.models import DirectPredictorConfig, TypedFactorDirectPredictor
from gdm_factor_diffusion.training import (
    DenoiserTrainer,
    LabeledDeploymentDataset,
    TrainerConfig,
    make_labeled_collator,
    sample_clean_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset = LabeledDeploymentDataset(
        root / "artifacts" / "datasets" / "phase1b-smoke",
        partitions=("train",),
    )
    collate = make_labeled_collator(dataset.feature_schema)
    labeled = collate([dataset[index] for index in range(len(dataset))])
    target = sample_clean_targets(labeled, mode="best")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    batch = labeled.factor_graph.to(device)
    clean_state = target.state.to(device)
    model = TypedFactorDirectPredictor.from_batch(
        batch,
        DirectPredictorConfig(hidden_dim=64, num_layers=2),
    ).to(device)
    trainer = DenoiserTrainer(
        model,
        CategoricalSchedule.linear(4, beta_end=0.2, device=device),
        torch.optim.AdamW(model.parameters(), lr=3e-3),
        TrainerConfig(capacity_weight=0.0, link_weight=0.0),
        model_kind="direct",
    )
    for _ in range(args.steps):
        trainer.train_step(batch, clean_state)
    terms = trainer.evaluate_step(batch, clean_state)
    accuracy = float(terms.clean_accuracy.item())
    if accuracy < 0.99:
        raise RuntimeError(f"Direct predictor failed overfit gate: accuracy={accuracy:.4f}")
    print(
        f"device={device} steps={args.steps} "
        f"accuracy={accuracy:.4f} loss={float(terms.clean_state.item()):.6f}"
    )


if __name__ == "__main__":
    main()
