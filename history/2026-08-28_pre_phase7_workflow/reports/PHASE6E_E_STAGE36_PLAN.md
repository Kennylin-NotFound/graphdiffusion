# Phase 6E-E Stage 3.6: Stochastic Efficiency Confirmation

Updated: 2026-06-22

Status: Stage 3.6A complete and frozen. No stochastic proposal count passed the
unchanged quality-and-time gate; confirmation is not authorized and Stages
3.6B--3.6D are closed without execution.

## 1. Evidence Boundary

The consumed Stage 3 pilot is immutable. It establishes a positive partial-
conditioning result and a stochastic quality signal, but stochastic `K=8`
fails the registered 1.10 total-time ratio. Stage 3 final data remains absent.

Stage 3.6 may use only `checkpoint_selection` to choose an inference budget.
It may not change the masked checkpoint, schedule, temperature, hard masks,
repair, fallback, objective evaluator, or gate thresholds.

## 2. Claim-to-Test Matrix

| Claim | Comparison | Primary evidence | Decision |
|---|---|---|---|
| A lower stochastic budget retains useful joint-choice diversity | deterministic `K=1` vs stochastic `K=2/4/8` | pre-fallback gap, paired wins/losses, raw feasibility | quality gate |
| The gain is available near deterministic cost | same comparison and timing scope | total and sampling time ratios | efficiency gate |
| The selected budget generalizes beyond its selection split | frozen budget on a fresh 128-instance contract | unchanged gates | confirmation gate |

## 3. Locked Selection Rule

All candidates use one batched graph per proposal chunk and eight conditional
model forwards. Random streams are coupled by instance across proposal counts.
A proposal count passes only if it:

1. improves mean pre-fallback gap by at least 5%;
2. has more paired wins than losses;
3. does not reduce raw feasibility or final verified success; and
4. stays within 1.10 times deterministic total online time.

Select the smallest passing proposal count. If none passes, stop the
diffusion-specific branch. Do not modify the rule after calibration.

## 4. Staged Execution

- **3.6A (complete):** implement, smoke-test, and run the locked 64-instance
  checkpoint-selection calibration.
- **3.6B (not authorized):** if 3.6A selects a budget, generate, label, audit, and freeze the
  pre-registered 128-instance confirmation contract.
- **3.6C (not authorized):** evaluate deterministic `K=1`, selected stochastic K, and the frozen
  direct control exactly once on confirmation.
- **3.6D (not authorized):** enter multi-seed training only if the unchanged confirmation gate
  passes; otherwise freeze Outcome B and revise the method claim boundary.

Observed calibration: K=2/K=4/K=8 obtain relative pre-fallback-gap changes of
-10.65%/+27.25%/+49.06% at total-time ratios 1.108/1.280/1.610. No candidate
passes every requirement, so `selected_method=null` and Outcome B is frozen.
