"""Small explicit trainer for clean-state denoising and constraint guidance."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gdm_factor_diffusion.diffusion.categorical_diffusion import q_sample
from gdm_factor_diffusion.diffusion.categorical_schedule import CategoricalSchedule
from gdm_factor_diffusion.graph.batch_adapter import FactorGraphBatch

from .objectives import ObjectiveTerms, compute_objective


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    capacity_weight: float = 0.1
    link_weight: float = 0.1
    gradient_clip_norm: float = 1.0

    def validate(self) -> None:
        if self.capacity_weight < 0 or self.link_weight < 0:
            raise ValueError("Guidance weights must be nonnegative.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")


class DenoiserTrainer:
    """Own one placement model, schedule, optimizer, and training-step contract."""

    def __init__(
        self,
        model: nn.Module,
        schedule: CategoricalSchedule,
        optimizer: torch.optim.Optimizer,
        config: TrainerConfig | None = None,
        *,
        model_kind: str = "diffusion",
    ) -> None:
        if model_kind not in {"diffusion", "direct"}:
            raise ValueError("model_kind must be 'diffusion' or 'direct'.")
        self.model = model
        self.schedule = schedule
        self.optimizer = optimizer
        self.config = config or TrainerConfig()
        self.config.validate()
        self.model_kind = model_kind
        self.step = 0

    def _predict(
        self,
        batch: FactorGraphBatch,
        clean_state: Tensor,
        timestep: Tensor | None,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        if self.model_kind == "direct":
            return self.model(batch)
        if timestep is None:
            raise ValueError("Diffusion training requires a timestep tensor.")
        noisy_state = q_sample(
            clean_state,
            timestep,
            batch.candidate_mask,
            self.schedule,
            batch.service_mask,
            generator=generator,
        )
        return self.model(batch, noisy_state, timestep)

    def train_step(
        self,
        batch: FactorGraphBatch,
        clean_state: Tensor,
        timestep: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self._predict(batch, clean_state, timestep, generator=generator)
        terms = compute_objective(
            logits,
            clean_state,
            batch,
            capacity_weight=self.config.capacity_weight,
            link_weight=self.config.link_weight,
        )
        terms.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.gradient_clip_norm,
        )
        self.optimizer.step()
        self.step += 1
        metrics = terms.detached_metrics()
        metrics["gradient_norm"] = float(gradient_norm.detach().item())
        metrics["step"] = float(self.step)
        return metrics

    @torch.no_grad()
    def evaluate_step(
        self,
        batch: FactorGraphBatch,
        clean_state: Tensor,
        timestep: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> ObjectiveTerms:
        self.model.eval()
        logits = self._predict(batch, clean_state, timestep, generator=generator)
        return compute_objective(
            logits,
            clean_state,
            batch,
            capacity_weight=self.config.capacity_weight,
            link_weight=self.config.link_weight,
        )


def save_checkpoint(
    path: str | Path,
    trainer: DenoiserTrainer,
    *,
    metadata: dict[str, Any],
    runtime_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically save model, optimizer, schedule, trainer state, and metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "schedule_betas": trainer.schedule.betas.detach().cpu(),
            "model_kind": trainer.model_kind,
            "trainer_config": asdict(trainer.config),
            "step": trainer.step,
            "metadata": metadata,
            "runtime_state": runtime_state or {},
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    trainer: DenoiserTrainer,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Restore a trusted local checkpoint into an already constructed trainer."""

    payload = restore_checkpoint(path, trainer, map_location=map_location)
    return dict(payload["metadata"])


def restore_checkpoint(
    path: str | Path,
    trainer: DenoiserTrainer,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Restore a checkpoint and return its complete trusted local payload."""

    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    checkpoint_model_kind = str(payload.get("model_kind", "diffusion"))
    if checkpoint_model_kind != trainer.model_kind:
        raise ValueError("Checkpoint and trainer model kinds disagree.")
    checkpoint_schedule = CategoricalSchedule.from_betas(
        payload["schedule_betas"].to(trainer.schedule.betas.device)
    )
    if checkpoint_schedule.num_steps != trainer.schedule.num_steps:
        raise ValueError("Checkpoint and trainer diffusion schedules have different sizes.")
    trainer.schedule = checkpoint_schedule
    trainer.model.load_state_dict(payload["model"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.step = int(payload["step"])
    return payload
