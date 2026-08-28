# Phase 6E-E Stage 3.7: Diffusion-Centered Matched-Time Test

Updated: 2026-06-22

Status: development gate complete and passed. The strict optimization preflight
selects the legacy decoder, Direct K=64 is the closest total-time baseline, and
all method-level matched-time checks pass. A new sealed multi-seed contract is
authorized but not yet generated or executed.

## 1. Paper Narrative

Direct prediction draws complete placements from a one-pass factorized choice
distribution. Increasing its proposal count broadens search but does not let
later service choices condition on committed placements and residual resources.
The proposed absorbing-MASK graph diffusion instead denoises a partial
deployment through a sequence of graph-conditioned completion steps. Its core
claim is therefore method-level:

> Under a comparable total online budget, structured reverse completion
> produces higher-quality verified deployments than additional independent
> proposals from a one-pass direct predictor.

The deterministic decoder is an ablation of the same conditional denoiser. It
tests the contribution of partial-state conditioning; it is not the primary
non-diffusion baseline. Stochastic K=8 tests the full diffusion-based solver and
is allowed to expose a quality-runtime tradeoff relative to deterministic K=1.

## 2. Claim-to-Test Matrix

| Claim | Comparison | Metrics | Acceptance |
|---|---|---|---|
| Full graph diffusion improves quality at comparable online cost | optimized stochastic K=8 vs time-matched retrained direct K | pre-fallback gap, verified gap, paired outcomes, raw feasibility, total time | time ratio 0.90--1.10; at least 10% pre-fallback-gap improvement; more wins than losses; final gap and raw feasibility not worse |
| Partial-state conditioning is an important mechanism | deterministic K=1 vs direct at its closest time budget | pre-fallback gap, raw feasibility, total time | diagnostic, already supported by the consumed pilot |
| Stochastic proposals provide an adjustable quality budget | stochastic K=8 vs deterministic K=1 | quality and time ratios | report tradeoff; no near-equal-time superiority claim |

## 3. Bounded Optimization Scope

Before the matched-time test, one implementation-only optimization pass is
allowed on `checkpoint_selection`:

- vectorize residual link-mask aggregation across particles and dependencies;
- batch service selection and categorical draws without changing the model,
  schedule, probabilities, hard masks, repair, fallback, or exact objective;
- keep the legacy decoder unchanged so frozen pilot evidence remains
  reproducible.

Acceptance requires exact deterministic replay against the legacy decoder,
deterministic stochastic replay within the new decoder for a fixed seed, finite
outputs, hard verification, and complete regression. If the optimized path is
not faster on a bounded smoke benchmark, retain the legacy path.

## 4. Checkpoint-Selection Experiment

- Data: the existing 64-instance `checkpoint_selection` split only.
- Model: frozen one-seed direct and masked checkpoints.
- Diffusion budget: stochastic K=8, eight reverse completion transitions.
- Direct grid: K in {32, 48, 64, 80, 96}; choose the total-time-closest K.
- Shared inference: identical repair budget, deterministic fallback, verifier,
  and exact weighted end-to-end latency evaluator.
- Timing: warm CUDA kernels first; record neural sampling and total online time.

This is a development gate, not final paper evidence. A positive result only
authorizes a new sealed, multi-seed contract; a negative result closes the
matched-time diffusion-superiority story.

Observed result: Diffusion K=8 versus Direct K=64 obtains a 1.095 total-time
ratio, 63.35% relative pre-fallback-gap improvement, and 42/10/12 paired
wins/losses/ties. Final gap, raw feasibility, and final success are not worse.

## 5. Stop Rules

- Do not access the consumed `pilot_gate` or any prior final partition.
- Do not change K, thresholds, checkpoints, or optimization semantics after
  reading the 64-instance aggregate.
- Do not rewrite Sections IV--V until a new sealed multi-seed evaluation exists.
- If the method-level matched-time gate fails, retain diffusion as the modeling
  framework but remove solver-superiority language.
