"""Independent trainer and checkpoint contract for the Stage 3 model family."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from gdm_factor_diffusion.diffusion import (
    AbsorbingMaskSchedule,
    PartialPlacementState,
    corrupt_with_absorbing_mask,
)
from gdm_factor_diffusion.graph import FactorGraphBatch

from .masked_objectives import MaskedObjectiveTerms, compute_masked_objective


@dataclass(frozen=True, slots=True)
class MaskedTrainerConfig:
    gradient_clip_norm: float = 1.0

    def validate(self) -> None:
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")


class MaskedConditionalTrainer:
    """Train one conditional model using hidden-service reconstruction only."""

    model_kind = "masked_conditional"

    def __init__(
        self,
        model: nn.Module,
        schedule: AbsorbingMaskSchedule,
        optimizer: torch.optim.Optimizer,
        config: MaskedTrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.optimizer = optimizer
        self.config = config or MaskedTrainerConfig()
        self.config.validate()
        self.step = 0

    def _sample_timestep(
        self,
        batch: FactorGraphBatch,
        generator: torch.Generator | None,
    ) -> Tensor:
        return torch.randint(
            1,
            self.schedule.num_steps + 1,
            (batch.batch_size,),
            device=batch.candidate_mask.device,
            generator=generator,
        )

    def _predict(
        self,
        batch: FactorGraphBatch,
        clean_state: Tensor,
        timestep: Tensor | None,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, PartialPlacementState, Tensor]:
        used_timestep = (
            self._sample_timestep(batch, generator) if timestep is None else timestep
        )
        partial = corrupt_with_absorbing_mask(
            clean_state,
            used_timestep,
            batch.candidate_mask,
            self.schedule,
            batch.service_mask,
            generator=generator,
        )
        return self.model(batch, partial, used_timestep), partial, used_timestep

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
        logits, partial, used_timestep = self._predict(
            batch, clean_state, timestep, generator
        )
        terms = compute_masked_objective(logits, clean_state, partial, batch)
        terms.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        self.step += 1
        metrics = terms.detached_metrics()
        metrics.update(
            {
                "gradient_norm": float(gradient_norm.detach().item()),
                "masked_fraction": float(
                    ((batch.service_mask & ~partial.committed_mask).sum() / batch.service_mask.sum()).item()
                ),
                "timestep_mean": float(used_timestep.float().mean().item()),
                "step": float(self.step),
            }
        )
        return metrics

    @torch.no_grad()
    def evaluate_step(
        self,
        batch: FactorGraphBatch,
        clean_state: Tensor,
        timestep: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> MaskedObjectiveTerms:
        self.model.eval()
        logits, partial, _ = self._predict(batch, clean_state, timestep, generator)
        return compute_masked_objective(logits, clean_state, partial, batch)


def save_masked_checkpoint(
    path: str | Path,
    trainer: MaskedConditionalTrainer,
    *,
    metadata: dict[str, Any],
    runtime_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically save a checkpoint that cannot be loaded as the old model family."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model_kind": trainer.model_kind,
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "mask_schedule": asdict(trainer.schedule),
            "trainer_config": asdict(trainer.config),
            "step": trainer.step,
            "metadata": metadata,
            "runtime_state": runtime_state or {},
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def restore_masked_checkpoint(
    path: str | Path,
    trainer: MaskedConditionalTrainer,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Restore a trusted local Stage 3 checkpoint and return its payload."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if payload.get("model_kind") != trainer.model_kind:
        raise ValueError("Checkpoint is not a masked_conditional model.")
    checkpoint_schedule = AbsorbingMaskSchedule(**payload["mask_schedule"])
    if checkpoint_schedule != trainer.schedule:
        raise ValueError("Checkpoint and trainer MASK schedules disagree.")
    trainer.model.load_state_dict(payload["model"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.step = int(payload["step"])
    return payload


def load_masked_checkpoint(
    path: str | Path,
    trainer: MaskedConditionalTrainer,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    return dict(
        restore_masked_checkpoint(path, trainer, map_location=map_location)["metadata"]
    )
