"""Time schedule for uniform-replacement categorical diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CategoricalSchedule:
    """Beta values and cumulative clean-state retention probabilities."""

    betas: Tensor
    alpha_bars: Tensor

    def __post_init__(self) -> None:
        if (
            self.betas.ndim != 1
            or self.betas.numel() < 1
            or not self.betas.is_floating_point()
        ):
            raise ValueError("betas must be a nonempty floating-point vector.")
        if (
            not torch.isfinite(self.betas).all()
            or (self.betas <= 0).any()
            or (self.betas >= 1).any()
        ):
            raise ValueError("Every beta must be finite and lie strictly in (0, 1).")
        if self.alpha_bars.shape != (self.betas.numel() + 1,):
            raise ValueError("alpha_bars must contain entries for timesteps 0 through T.")
        if (
            self.alpha_bars.dtype != self.betas.dtype
            or self.alpha_bars.device != self.betas.device
        ):
            raise ValueError("alpha_bars must share the beta dtype and device.")
        expected = torch.cat(
            (
                torch.ones(
                    1, dtype=self.betas.dtype, device=self.betas.device
                ),
                torch.cumprod(1.0 - self.betas, dim=0),
            )
        )
        if not torch.allclose(self.alpha_bars, expected):
            raise ValueError("alpha_bars is inconsistent with betas.")

    @classmethod
    def from_betas(cls, betas: Tensor) -> "CategoricalSchedule":
        values = torch.as_tensor(betas)
        if values.ndim != 1 or values.numel() < 1 or not values.is_floating_point():
            raise ValueError("betas must be a nonempty floating-point vector.")
        if not torch.isfinite(values).all() or (values <= 0).any() or (values >= 1).any():
            raise ValueError("Every beta must be finite and lie strictly in (0, 1).")
        values = values.detach().clone()
        alpha_bars = torch.cat(
            (
                torch.ones(1, dtype=values.dtype, device=values.device),
                torch.cumprod(1.0 - values, dim=0),
            )
        )
        return cls(betas=values, alpha_bars=alpha_bars)

    @classmethod
    def linear(
        cls,
        num_steps: int,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 0.2,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> "CategoricalSchedule":
        """Build a simple monotone schedule; tune values only after Phase 3."""

        if num_steps < 1:
            raise ValueError("num_steps must be positive.")
        return cls.from_betas(
            torch.linspace(
                beta_start,
                beta_end,
                num_steps,
                dtype=dtype,
                device=device,
            )
        )

    @property
    def num_steps(self) -> int:
        return int(self.betas.numel())

    @property
    def terminal_retention(self) -> float:
        return float(self.alpha_bars[-1].item())

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "CategoricalSchedule":
        return CategoricalSchedule.from_betas(self.betas.to(device=device, dtype=dtype))

    def normalize_timesteps(
        self,
        timestep: int | Tensor,
        *,
        batch_size: int,
        device: torch.device,
        allow_zero: bool = False,
    ) -> Tensor:
        """Return one timestep per graph in a batch."""

        value = torch.as_tensor(timestep, dtype=torch.long, device=device)
        if value.ndim == 0:
            value = value.expand(batch_size)
        if value.shape != (batch_size,):
            raise ValueError("timestep must be a scalar or a vector with shape [B].")
        minimum = 0 if allow_zero else 1
        if (value < minimum).any() or (value > self.num_steps).any():
            raise ValueError(
                f"timestep must lie in [{minimum}, {self.num_steps}]."
            )
        return value

    def beta_at(self, timestep: Tensor, *, dtype: torch.dtype) -> Tensor:
        return self.betas.to(device=timestep.device, dtype=dtype)[timestep - 1]

    def alpha_bar_at(self, timestep: Tensor, *, dtype: torch.dtype) -> Tensor:
        return self.alpha_bars.to(device=timestep.device, dtype=dtype)[timestep]
