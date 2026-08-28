"""Training and checkpoint utilities for sequential conditional GNNs."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gdm_factor_diffusion.graph import FactorGraphBatch

from .sequential_objectives import (
    SequentialObjectiveTerms,
    build_teacher_forced_prefix,
    compute_sequential_objective,
)


@dataclass(frozen=True, slots=True)
class SequentialTrainerConfig:
    gradient_clip_norm: float = 1.0

    def validate(self) -> None:
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")


class SequentialConditionalTrainer:
    """Train an autoregressive GNN policy with teacher-forced prefixes."""

    model_kind = "sequential_conditional"

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: SequentialTrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config or SequentialTrainerConfig()
        self.config.validate()
        self.step = 0

    def _sample_step(
        self,
        batch: FactorGraphBatch,
        generator: torch.Generator | None,
    ) -> Tensor:
        active_count = batch.service_mask.sum(dim=1)
        sampled = torch.empty(
            batch.batch_size,
            dtype=torch.long,
            device=batch.candidate_mask.device,
        )
        for batch_index, count in enumerate(active_count.tolist()):
            sampled[batch_index] = torch.randint(
                int(count),
                (1,),
                device=batch.candidate_mask.device,
                generator=generator,
            )[0]
        return sampled

    def _predict(
        self,
        batch: FactorGraphBatch,
        clean_state: Tensor,
        order: Tensor,
        step_index: Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        used_step = (
            self._sample_step(batch, generator)
            if step_index is None
            else torch.as_tensor(
                step_index,
                dtype=torch.long,
                device=batch.candidate_mask.device,
            )
        )
        partial, target_service, step_fraction = build_teacher_forced_prefix(
            clean_state,
            order,
            used_step,
            batch,
        )
        logits = self.model(batch, partial, target_service, step_fraction)
        return logits, target_service, used_step, partial

    def train_step(
        self,
        batch: FactorGraphBatch,
        clean_state: Tensor,
        order: Tensor,
        step_index: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits, target_service, used_step, partial = self._predict(
            batch,
            clean_state,
            order,
            step_index,
            generator,
        )
        terms = compute_sequential_objective(
            logits,
            clean_state,
            partial,
            target_service,
            batch,
        )
        terms.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
        )
        self.optimizer.step()
        self.step += 1
        metrics = terms.detached_metrics()
        metrics.update(
            {
                "gradient_norm": float(gradient_norm.detach().item()),
                "prefix_fraction_mean": float(
                    (
                        partial.committed_mask.sum(dim=1).float()
                        / batch.service_mask.sum(dim=1).float().clamp_min(1.0)
                    )
                    .mean()
                    .item()
                ),
                "step_index_mean": float(used_step.float().mean().item()),
                "step": float(self.step),
            }
        )
        return metrics

    @torch.no_grad()
    def evaluate_step(
        self,
        batch: FactorGraphBatch,
        clean_state: Tensor,
        order: Tensor,
        step_index: Tensor,
    ) -> SequentialObjectiveTerms:
        self.model.eval()
        logits, target_service, _, partial = self._predict(
            batch,
            clean_state,
            order,
            step_index,
            generator=None,
        )
        return compute_sequential_objective(
            logits,
            clean_state,
            partial,
            target_service,
            batch,
        )


def save_sequential_checkpoint(
    path: str | Path,
    trainer: SequentialConditionalTrainer,
    *,
    metadata: dict[str, Any],
    runtime_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically save a sequential-policy checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model_kind": trainer.model_kind,
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "trainer_config": asdict(trainer.config),
            "step": trainer.step,
            "metadata": metadata,
            "runtime_state": runtime_state or {},
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def restore_sequential_checkpoint(
    path: str | Path,
    trainer: SequentialConditionalTrainer,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Restore a trusted local sequential-policy checkpoint."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("model_kind") != trainer.model_kind:
        raise ValueError("Checkpoint is not a sequential_conditional model.")
    trainer.model.load_state_dict(payload["model"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.step = int(payload["step"])
    return payload


def load_sequential_checkpoint(
    path: str | Path,
    trainer: SequentialConditionalTrainer,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    return dict(
        restore_sequential_checkpoint(path, trainer, map_location=map_location)[
            "metadata"
        ]
    )
