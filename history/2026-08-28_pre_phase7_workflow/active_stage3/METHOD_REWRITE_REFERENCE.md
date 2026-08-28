# Stage 3 Method Rewrite Reference

Updated: 2026-06-22

Purpose: implementation-grounded reference for rewriting the manuscript after
the Stage 3.5 evidence gates. The pilot is now frozen as Outcome B; this file
records the implementation contract but does not authorize claims beyond that
evidence boundary.

## 1. Problem-to-Method Alignment

Section II minimizes weighted application end-to-end latency over a categorical
service-to-device placement subject to exactly-one deployment, compatibility,
device capacity, and direct-link feasibility. The learned solver must therefore
represent deployment-level dependencies while returning a complete assignment
that is checked against the unchanged exact verifier and latency evaluator.

The previous denoiser predicted all service locations simultaneously from a
fully categorical noisy assignment. Diagnostics showed useful intermediate
predictions, but neither the old loss nor the final-only sampler established a
diffusion-specific advantage over a matched-time direct predictor. Stage 3
changes the generative state and conditioning contract rather than enlarging
the graph network indiscriminately.

## 2. Partial Deployment State

For instance graph `H`, define the reverse state at step `t` as

`S_t = (A_t, C_t)`,

where `A_t[v]` is the committed device of service `v` and `C_t[v]` indicates
whether that assignment is visible. An uncommitted active service has
`A_t[v] = -1`; padding is distinguished by the separate service mask. MASK is
an input state, not an additional physical device or an output category.

This representation exposes residual resource and dependency information after
each commitment. It directly addresses the deployment coupling that a
one-shot factorized output can underuse.

## 3. Absorbing-MASK Corruption

The mask probability is

`rho_t = (t / T)^gamma`, with `rho_0 = 0` and `rho_T = 1`.

Training samples `t` uniformly from `{1,...,T}` and independently hides active
clean assignments according to `rho_t`; every positive timestep hides at least
one service. Once hidden in the forward process, a category is represented only
by MASK. The locked pilot uses `T = 8` and `gamma = 1`.

## 4. Typed Partial-State Context

The static typed factor graph remains unchanged: service, device, dependency,
and application nodes encode demands, capacities, compatibility, DAG incidence,
and application membership. Stage 3 adds state-dependent features:

- service context `[committed, selected processing latency]`;
- device context `[normalized committed load, normalized residual capacity]`
  for every resource;
- dependency context `[source visible, target visible, selected pair latency,
  selected pair admissible]`;
- a learned MASK embedding for uncommitted services and a selected-device
  embedding for committed services.

Unknown endpoint pairs contribute no fabricated latency/admissibility value.
The context is recomputed from the partial assignment before each model call.

## 5. Conditional Typed Graph Denoiser

Static node embeddings, partial-state context, selected-device/MASK state, and
timestep embeddings are combined before relation-specific typed factor message
passing. The output remains compatible-device logits of shape `[B,M,D]`:

`p_theta(x_v | S_t, H, t)`.

Although the head returns per-service categorical logits, sequential
conditioning changes after commitments, so subsequent predictions depend on
the previously selected joint deployment prefix. The model has 920,833
parameters; the matched direct control has 901,505 parameters. This small
parameter difference must be reported with online runtime.

## 6. Training Objective and Supervision

Let `U_t` be hidden active services. The Stage 3 pilot objective is

`L_mask = - sum_{v in U_t} log p_theta(x_v^* | S_t, H, t) / |U_t|`.

Visible services do not contribute reconstruction loss. Soft capacity/link
guidance and energy-weighted sampling are disabled in the structural pilot,
because prior ablations did not establish their necessity and retaining them
would confound the state redesign. Supervision is the best verified placement
from each equivalent-MILP pool. The direct control uses the same best-only
targets, batches, optimizer, graph width/depth, and 20,000-step budget.

This design tests conditional generation; it does not yet prove that stochastic
diffusion is superior. That claim is delegated to the deterministic-versus-
stochastic decoder comparison.

## 7. Shared Reverse Completion

Both decoders start from all MASK and follow the same reverse unmask schedule.
Before committing service `v` to device `i`, a hard residual mask removes
choices violating:

1. service-device compatibility;
2. device capacity after adding `v` to committed load;
3. direct-link admissibility to every already committed dependency neighbor.

The deterministic decoder commits the highest-confidence admissible category
with deterministic tie breaking. The stochastic decoder samples admissible
categories from the same logits and schedule for multiple particles. They use
the same number of reverse transitions, verifier, bounded repair, fallback, and
exact latency ranking. An unresolved service remains MASK; finite transitions
provide a complete stopping condition and an explicit incomplete outcome.

## 8. Feasibility and Objective Semantics

Neural prediction does not claim to guarantee global feasibility or optimality.
Hard residual masking prevents violations against the committed prefix, while
the unchanged verifier is the final authority. Incomplete/infeasible proposals
may enter bounded repair; deterministic fallback is invoked under the existing
budget. Only verified complete placements enter exact weighted end-to-end
latency evaluation and final minimum-latency selection.

Checkpoint ranking prioritizes pre-fallback success and pre-fallback gap before
raw feasibility and online time. This prevents fallback from hiding a weak
learned generator.

## 9. Evidence and Claim Gates

The paper may claim partial-conditioning value only if deterministic conditional
completion passes Gate R3-B against the retrained direct predictor. A
diffusion-specific claim additionally requires stochastic completion to pass
Gate R3-C against the deterministic decoder using the same conditional model.

Possible manuscript outcomes:

- Outcome A: both gates pass; retain graph diffusion as the core solver.
- Outcome B: conditioning helps, but stochastic decoding does not satisfy the
  complete quality-and-time gate; present a typed graph conditional placement
  generator and treat stochastic decoding as a quality-runtime option.
- Outcome C: direct prediction remains stronger; remove solver-superiority and
  diffusion-necessity claims.

Stage 3.6 closes the mechanism-level efficiency branch with no stochastic K
passing the deterministic-decoder gate. Thus, stochastic completion remains a
quality--runtime option relative to deterministic conditional completion.
Stage 3.7 answers a separate method-level question: on checkpoint-selection,
the full Masked Diffusion method at K=8 outperforms the one-pass Direct baseline
at a comparable total online budget. This positive result authorizes a newly
sealed multi-seed evaluation, but it is not final manuscript evidence.

## 10. Recommended Section IV Rewrite Order

1. Framework overview and categorical partial state.
2. Typed graph representation and partial deployment context.
3. Absorbing-MASK corruption and masked reconstruction training.
4. Conditional typed factor denoiser.
5. Deterministic and stochastic reverse completion with hard masks.
6. Verification, repair, fallback, and exact final selection.

The training figure should replace the previous energy-weighted target flow
with best verified MILP supervision and masked conditional reconstruction. The
inference figure should distinguish the shared conditional model from the two
decoding policies and retain the explicit incomplete/failure exit.

## 11. Notation and Wording Risks

- Do not call `-1` a device category; it denotes uncommitted/padded positions
  under separate masks.
- Do not describe hard residual masking as a complete feasibility guarantee.
- Do not retain soft-guidance or energy-weighted-loss equations as active Stage
  3 training components unless later experiments explicitly reintroduce them.
- Do not claim joint-distribution learning solely from sequential conditioning;
  state that the conditional factorization can represent prefix-dependent
  correlations and test its practical effect empirically.
- Do not claim optimality for MILP pools unless the solver status proves it;
  use best verified placement where appropriate.
