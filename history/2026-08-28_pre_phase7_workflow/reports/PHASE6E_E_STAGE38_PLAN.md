# Phase 6E-E Stage 3.8: Sealed Multi-Seed Confirmation

Updated: 2026-06-22

## Objective

Test whether the Stage 3.7 development result replicates on a fresh sealed ID
partition and across three training seeds. The primary paper claim is:

> Under a comparable total online budget, absorbing-MASK graph diffusion
> produces higher-quality verified deployments than additional independent
> proposals from a one-pass direct predictor.

Deterministic masked completion remains a mechanism ablation. It is not the
time-matched non-diffusion baseline and does not define the primary claim.

## Claim-to-Test Matrix

| Claim | Comparison | Evidence | Acceptance |
|---|---|---|---|
| Method-level quality at comparable online cost | Masked Diffusion K=8 vs Direct K=64 | Three seeds x 128 sealed ID instances | Every seed has a 0.90--1.10 time ratio; aggregate pre-fallback gap improves by at least 10%; improvement is positive for every seed |
| Advantage is not a fallback artifact | Same comparison before fallback | Pre-fallback success, gap, raw feasibility, and instance-level paired ranking | Success and raw feasibility do not decrease; paired wins exceed losses with two-sided sign-test p <= 0.05 |
| Final solver quality is preserved | Same verified post-processing | Final verified success and exact gap to pool best | Final success does not decrease and final gap is not worse |
| Partial-state conditioning contributes independently | Deterministic masked K=1 vs Direct K=64 | Descriptive ablation under the same seeds and instances | Reported separately; it does not gate the primary diffusion claim |

## Frozen Contract

- Training data and checkpoint selection reuse the frozen Stage 3 development
  partitions. No pilot or prior final partition is reopened.
- Seeds are `2026070111`, `2026070112`, and `2026070113`. The first checkpoint
  pair is reused; the other two pairs use the unchanged 20,000-step protocol.
- A fresh 128-instance `sealed_test_id` partition uses the same ID generation
  distribution and a new base seed `2026070601`.
- Direct K=64 and Masked Diffusion K=8 are fixed before sealed-data generation.
- Repair, deterministic fallback, temperature, batching, and verifier settings
  are unchanged from Stage 3.7.
- Metrics are reported per seed, across all seed-instance records, and by 128
  independent instance blocks. Mean and standard deviation across seeds are
  retained for manuscript reporting.

## Stop Rules

- If either new training seed fails preflight or checkpoint-freeze validation,
  sealed data must remain absent.
- Once `sealed_test_id` is generated, no architecture, proposal count,
  checkpoint-selection rule, threshold, or post-processing budget may change.
- If any required acceptance check fails, the broad diffusion-superiority
  claim is rejected. The result must be reported as a limitation or restricted
  to the narrower development observation.
- The sealed evaluation is executed once and is resumable only at immutable
  seed-instance record boundaries.

## Execution Stages

1. Freeze preparation lock and pass four CUDA training preflights.
2. Train Direct and Masked models for seeds `2026070112` and `2026070113`.
3. Validate all six checkpoint families and freeze the multi-seed training set.
4. Generate, label, audit, and freeze the sealed 128-instance dataset.
5. Evaluate all three methods for all seeds, aggregate once, and freeze the
   final decision.

Current status: preparation lock and all four CUDA preflights are frozen and
verified. The two new training pairs are authorized; sealed data do not exist.
