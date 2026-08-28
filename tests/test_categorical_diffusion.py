import numpy as np
import pytest
import torch

from gdm_factor_diffusion.diffusion import (
    CategoricalSchedule,
    build_reverse_timestep_grid,
    masked_softmax,
    model_posterior,
    model_posterior_between,
    p_sample,
    q_posterior,
    q_probabilities,
    q_sample,
    reverse_sample_loop,
    sample_prior,
    state_to_one_hot,
    validate_state,
)


def _variable_batch(device: str = "cpu"):
    candidate_mask = torch.tensor(
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, False, False, False],
            ],
            [
                [False, True, True, True],
                [False, False, True, False],
                [True, False, True, False],
            ],
        ],
        device=device,
    )
    clean_state = torch.tensor(
        [[0, 1, -1], [2, 2, 0]],
        dtype=torch.long,
        device=device,
    )
    return candidate_mask, clean_state


def test_masked_probabilities_support_singleton_variable_candidates_and_padding() -> None:
    candidate_mask, state = _variable_batch()
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    probability = masked_softmax(logits, candidate_mask)

    assert torch.equal(validate_state(state, candidate_mask), candidate_mask.any(-1))
    assert torch.all(probability[~candidate_mask] == 0)
    assert torch.allclose(
        probability.sum(-1)[candidate_mask.any(-1)],
        torch.ones(5),
    )
    assert torch.all(probability.sum(-1)[~candidate_mask.any(-1)] == 0)

    one_hot = state_to_one_hot(state, candidate_mask)
    assert torch.all(one_hot.sum(-1)[candidate_mask.any(-1)] == 1)
    assert torch.all(one_hot.sum(-1)[~candidate_mask.any(-1)] == 0)

    prior_state = sample_prior(candidate_mask)
    validate_state(prior_state, candidate_mask)
    assert prior_state[0, 0].item() == 0
    assert prior_state[1, 1].item() == 2
    assert prior_state[0, 2].item() == -1


def test_schedule_and_forward_marginal_match_uniform_replacement_formula() -> None:
    schedule = CategoricalSchedule.from_betas(
        torch.tensor([0.25, 0.20], dtype=torch.float64)
    )
    candidate_mask, clean_state = _variable_batch()
    probability = q_probabilities(
        clean_state,
        timestep=2,
        candidate_mask=candidate_mask,
        schedule=schedule,
        dtype=torch.float64,
    )

    assert schedule.num_steps == 2
    assert schedule.terminal_retention == pytest.approx(0.6)
    assert probability[0, 0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert probability[0, 1].tolist() == pytest.approx([0.2, 0.8, 0.0, 0.0])
    assert probability[0, 2].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert probability[1, 0].tolist() == pytest.approx(
        [0.0, 0.4 / 3.0, 0.6 + 0.4 / 3.0, 0.4 / 3.0]
    )


def test_forward_sampling_empirical_frequency_matches_analytical_marginal() -> None:
    samples = 60_000
    schedule = CategoricalSchedule.from_betas(torch.tensor([0.3, 0.4]))
    candidate_mask = torch.ones((samples, 1, 4), dtype=torch.bool)
    clean_state = torch.zeros((samples, 1), dtype=torch.long)
    generator = torch.Generator().manual_seed(123)

    noisy_state = q_sample(
        clean_state,
        timestep=2,
        candidate_mask=candidate_mask,
        schedule=schedule,
        generator=generator,
    )
    empirical = torch.bincount(noisy_state[:, 0], minlength=4).float() / samples
    expected = q_probabilities(
        clean_state[:1],
        timestep=2,
        candidate_mask=candidate_mask[:1],
        schedule=schedule,
    )[0, 0]
    assert empirical.tolist() == pytest.approx(expected.tolist(), abs=0.007)


def test_exact_posterior_matches_direct_bayes_calculation() -> None:
    schedule = CategoricalSchedule.from_betas(
        torch.tensor([0.2, 0.35], dtype=torch.float64)
    )
    candidate_mask = torch.tensor([[[True, True, True]]])
    clean_state = torch.tensor([[0]])
    noisy_state = torch.tensor([[2]])
    posterior = q_posterior(
        noisy_state,
        clean_state,
        timestep=2,
        candidate_mask=candidate_mask,
        schedule=schedule,
        dtype=torch.float64,
    )[0, 0]

    beta = 0.35
    alpha_bar_previous = 0.8
    q_previous = np.asarray(
        [alpha_bar_previous + (1 - alpha_bar_previous) / 3, 0.2 / 3, 0.2 / 3]
    )
    likelihood = np.asarray([beta / 3, beta / 3, 1 - beta + beta / 3])
    expected = q_previous * likelihood
    expected /= expected.sum()
    assert posterior.tolist() == pytest.approx(expected.tolist())
    assert posterior.sum().item() == pytest.approx(1.0)


def test_model_posterior_reduces_to_exact_for_one_hot_clean_prediction() -> None:
    schedule = CategoricalSchedule.from_betas(torch.tensor([0.2, 0.35]))
    candidate_mask, clean_state = _variable_batch()
    noisy_state = q_sample(
        clean_state,
        timestep=torch.tensor([1, 2]),
        candidate_mask=candidate_mask,
        schedule=schedule,
        generator=torch.Generator().manual_seed(11),
    )
    clean_probability = state_to_one_hot(clean_state, candidate_mask)

    exact = q_posterior(
        noisy_state,
        clean_state,
        timestep=torch.tensor([1, 2]),
        candidate_mask=candidate_mask,
        schedule=schedule,
    )
    modeled = model_posterior(
        clean_probability,
        noisy_state,
        timestep=torch.tensor([1, 2]),
        candidate_mask=candidate_mask,
        schedule=schedule,
    )
    assert torch.allclose(modeled, exact, atol=1e-6)


def test_model_posterior_at_first_step_equals_predicted_clean_distribution() -> None:
    schedule = CategoricalSchedule.from_betas(torch.tensor([0.25, 0.3]))
    candidate_mask, clean_state = _variable_batch()
    noisy_state = q_sample(
        clean_state,
        timestep=1,
        candidate_mask=candidate_mask,
        schedule=schedule,
        generator=torch.Generator().manual_seed(5),
    )
    logits = torch.randn(candidate_mask.shape, generator=torch.Generator().manual_seed(9))
    clean_probability = masked_softmax(logits, candidate_mask)
    posterior = model_posterior(
        clean_probability,
        noisy_state,
        timestep=1,
        candidate_mask=candidate_mask,
        schedule=schedule,
    )
    assert torch.allclose(posterior, clean_probability, atol=1e-6)


def test_generalized_posterior_matches_adjacent_posterior() -> None:
    schedule = CategoricalSchedule.linear(8, beta_end=0.3)
    candidate_mask, clean_state = _variable_batch()
    noisy_state = q_sample(
        clean_state,
        timestep=6,
        candidate_mask=candidate_mask,
        schedule=schedule,
        generator=torch.Generator().manual_seed(20),
    )
    logits = torch.randn(candidate_mask.shape, generator=torch.Generator().manual_seed(21))
    clean_probability = masked_softmax(logits, candidate_mask)

    adjacent = model_posterior(
        clean_probability, noisy_state, 6, candidate_mask, schedule
    )
    generalized = model_posterior_between(
        clean_probability, noisy_state, 6, 5, candidate_mask, schedule
    )

    assert torch.allclose(adjacent, generalized, atol=1e-6)


def test_generalized_posterior_matches_manual_skipped_transition() -> None:
    schedule = CategoricalSchedule.linear(6, beta_end=0.3)
    candidate_mask = torch.ones((1, 1, 3), dtype=torch.bool)
    noisy_state = torch.tensor([[2]])
    clean_probability = torch.tensor([[[0.0, 1.0, 0.0]]])
    posterior = model_posterior_between(
        clean_probability,
        noisy_state,
        timestep=6,
        previous_timestep=2,
        candidate_mask=candidate_mask,
        schedule=schedule,
    )

    alpha_s = schedule.alpha_bars[2]
    alpha_t = schedule.alpha_bars[6]
    clean_at_s = torch.full((3,), (1.0 - alpha_s) / 3.0)
    clean_at_s[1] += alpha_s
    jump_retention = alpha_t / alpha_s
    observation_likelihood = torch.full((3,), (1.0 - jump_retention) / 3.0)
    observation_likelihood[2] += jump_retention
    manual = clean_at_s * observation_likelihood
    manual = manual / manual.sum()

    assert torch.allclose(posterior[0, 0], manual, atol=1e-6)


def test_reverse_timestep_grid_is_strict_and_has_requested_transitions() -> None:
    assert build_reverse_timestep_grid(100, 4) == (100, 75, 50, 25, 0)
    grid = build_reverse_timestep_grid(7, 3)
    assert len(grid) == 4
    assert grid[0] == 7
    assert grid[-1] == 0
    assert all(current > previous for current, previous in zip(grid, grid[1:]))


def test_reverse_sampling_respects_candidates_and_padding() -> None:
    schedule = CategoricalSchedule.linear(8, beta_end=0.4)
    candidate_mask, clean_state = _variable_batch()
    noisy_state = q_sample(
        clean_state,
        timestep=8,
        candidate_mask=candidate_mask,
        schedule=schedule,
        generator=torch.Generator().manual_seed(3),
    )
    logits = torch.randn(candidate_mask.shape, generator=torch.Generator().manual_seed(4))
    previous_state = p_sample(
        logits,
        noisy_state,
        timestep=8,
        candidate_mask=candidate_mask,
        schedule=schedule,
        generator=torch.Generator().manual_seed(6),
    )
    validate_state(previous_state, candidate_mask)


def test_complete_reverse_loop_accepts_a_model_independent_logits_callable() -> None:
    schedule = CategoricalSchedule.linear(6, beta_end=0.4)
    candidate_mask, target = _variable_batch()

    def oracle_logits(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del state, timestep
        logits = torch.full(candidate_mask.shape, -100.0)
        logits.scatter_(-1, target.clamp_min(0).unsqueeze(-1), 100.0)
        return logits.masked_fill(~candidate_mask, -torch.inf)

    sampled = reverse_sample_loop(
        oracle_logits,
        candidate_mask,
        schedule,
        generator=torch.Generator().manual_seed(12),
    )
    assert torch.equal(sampled, target)

    skipped = reverse_sample_loop(
        oracle_logits,
        candidate_mask,
        schedule,
        timesteps=build_reverse_timestep_grid(schedule.num_steps, 2),
        generator=torch.Generator().manual_seed(12),
    )
    assert torch.equal(skipped, target)


def test_reverse_callback_is_observational_and_preserves_rng_result() -> None:
    schedule = CategoricalSchedule.linear(6, beta_end=0.4)
    candidate_mask, _ = _variable_batch()
    logits = torch.randn(candidate_mask.shape, generator=torch.Generator().manual_seed(4))

    def fixed_logits(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del state, timestep
        return logits.masked_fill(~candidate_mask, -torch.inf)

    baseline = reverse_sample_loop(
        fixed_logits,
        candidate_mask,
        schedule,
        generator=torch.Generator().manual_seed(12),
    )
    records: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    observed = reverse_sample_loop(
        fixed_logits,
        candidate_mask,
        schedule,
        generator=torch.Generator().manual_seed(12),
        step_callback=lambda t, s, zt, clean, zs: records.append(
            (t, s, zt, clean, zs)
        ),
    )

    assert torch.equal(observed, baseline)
    assert len(records) == schedule.num_steps
    assert records[0][0:2] == (schedule.num_steps, schedule.num_steps - 1)
    assert records[-1][0:2] == (1, 0)
    records[0][4].fill_(0)
    assert torch.equal(observed, baseline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
def test_cuda_forward_posterior_and_reverse_sampling() -> None:
    candidate_mask, clean_state = _variable_batch("cuda")
    schedule = CategoricalSchedule.linear(10, beta_end=0.3).to("cuda")
    noisy_state = q_sample(clean_state, 7, candidate_mask, schedule)
    logits = torch.randn(candidate_mask.shape, device="cuda", requires_grad=True)
    clean_probability = masked_softmax(logits, candidate_mask)
    posterior = model_posterior(
        clean_probability, noisy_state, 7, candidate_mask, schedule
    )
    previous_state = p_sample(logits, noisy_state, 7, candidate_mask, schedule)

    validate_state(previous_state, candidate_mask)
    loss = posterior.square().sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
