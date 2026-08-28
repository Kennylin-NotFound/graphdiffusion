"""Run empirical and CUDA smoke checks for the Phase 2 diffusion core."""

from __future__ import annotations

import torch

from gdm_factor_diffusion.diffusion import (
    CategoricalSchedule,
    model_posterior,
    p_sample,
    q_probabilities,
    q_sample,
    validate_state,
)


def main() -> None:
    schedule = CategoricalSchedule.linear(100, beta_end=0.2)
    samples = 100_000
    candidate_mask = torch.ones((samples, 1, 4), dtype=torch.bool)
    clean_state = torch.zeros((samples, 1), dtype=torch.long)
    generator = torch.Generator().manual_seed(20260612)
    noisy_state = q_sample(
        clean_state,
        50,
        candidate_mask,
        schedule,
        generator=generator,
    )
    empirical = torch.bincount(noisy_state[:, 0], minlength=4).float() / samples
    analytical = q_probabilities(
        clean_state[:1], 50, candidate_mask[:1], schedule
    )[0, 0]
    maximum_error = float((empirical - analytical).abs().max())
    if maximum_error > 0.01:
        raise RuntimeError(f"Forward empirical error is too large: {maximum_error}")
    print(
        f"forward_max_error={maximum_error:.6g} "
        f"terminal_retention={schedule.terminal_retention:.6g}"
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        mask = torch.tensor(
            [[[True, True, False], [False, True, True]]],
            device=device,
        )
        clean = torch.tensor([[0, 2]], device=device)
        cuda_schedule = schedule.to(device)
        noisy = q_sample(clean, 80, mask, cuda_schedule)
        logits = torch.randn(mask.shape, device=device, requires_grad=True)
        probability = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        posterior = model_posterior(probability, noisy, 80, mask, cuda_schedule)
        previous = p_sample(logits, noisy, 80, mask, cuda_schedule)
        validate_state(previous, mask)
        posterior.square().sum().backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise RuntimeError("CUDA reverse posterior produced invalid gradients.")
        print(
            f"cuda_device={torch.cuda.get_device_name(0)} "
            f"posterior_shape={tuple(posterior.shape)}"
        )


if __name__ == "__main__":
    main()
