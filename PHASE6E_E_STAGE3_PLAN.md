# Phase 6E-E Stage 3: Masked Partial-Assignment Diffusion

Updated: 2026-06-21

## 1. Decision and Scope

Stage 2B confirms that the current reverse trajectory contains useful
deployment candidates, but the rescued solver still trails direct K=96. Stage
3 therefore changes the diffusion state and generation contract while
preserving the optimization problem, typed factor graph, exact evaluator, hard
verifier, bounded repair, deterministic fallback, and all frozen evidence.

The redesign is not a generic larger GNN. It is an absorbing-MASK conditional
completion model whose decisions can use committed placements, residual device
capacity, and the link feasibility of already placed dependency neighbors.

Stage 3 must answer two separate questions:

1. Does partial-assignment conditioning improve over one-shot prediction?
2. Does stochastic masked diffusion improve over deterministic conditional
   completion under matched online time?

No final-ID, shift, or scale artifact from Phases 6D--6E may be used for Stage
3 architecture, checkpoint, proposal-budget, or schedule selection.

## 2. Claim-to-Test Matrix

| Candidate claim | Required comparison | Primary metrics | Decision |
|---|---|---|---|
| Partial conditioning helps deployment decisions | Retrained one-shot direct vs deterministic masked completion | raw-any feasibility, pre-fallback gap, verified gap, time | Gate R3-B |
| Stochastic diffusion adds value beyond conditioning | Deterministic vs stochastic decoding from the same masked model and commit schedule | paired objective outcomes, matched-time gap, raw feasibility | Gate R3-C |
| The redesign preserves exact problem semantics | Every accepted output through the existing verifier/evaluator | verified success, exact objective agreement | Engineering gate |
| Improvement is reproducible | Three independent training seeds after the pilot | mean/std, paired outcomes, checkpoint hashes | Multi-seed freeze |
| The final method generalizes | One newly sealed Stage 3 final contract after all choices are frozen | ID/shift/scale quality and runtime | Manuscript decision |

The frozen Phase 6D direct K=96 and Phase 6E-E rescue remain historical
anchors. A newly trained one-shot direct control using the same Stage 3 data,
targets, step budget, and backbone is mandatory for causal fairness.

## 3. Locked Technical Design

### 3.1 Partial Placement State

Introduce an explicit state object rather than overloading padding value `-1`:

- `assignment[B,M]`: committed local device index, `-1` otherwise;
- `committed_mask[B,M]`: active services whose assignments are visible;
- the existing `service_mask` continues to identify real versus padded
  services.

Validation requires every committed assignment to be compatible, every
uncommitted active service to use `-1`, and every padded service to remain
uncommitted. Model output remains `[B,M,D]` device logits; MASK is an input
state, not an extra deployment device.

### 3.2 Absorbing-MASK Corruption

Use a monotone mask schedule `rho_t` with `rho_0=0` and `rho_T=1`. At training
time, a clean verified placement is corrupted by hiding each active service
according to `rho_t`; at least one service is hidden for every positive
timestep. The reconstruction loss is evaluated only on hidden active services:

`L_mask = CE(z0[hidden], p_theta(z0 | visible(z0,t), H)[hidden])`.

The Stage 3 pilot uses this masked conditional cross-entropy alone. Capacity
and link guidance remain configurable but disabled, because their prior
evidence is mixed and adding them would confound the structural test.

### 3.3 Typed Graph Context

Preserve the static typed factor graph and add a separate partial-state context
builder:

- service context: committed flag and selected processing latency when known;
- device context: load and residual-capacity ratios from committed services;
- dependency context: endpoint visibility, and selected pair latency/
  admissibility only when both endpoints are committed;
- a learned MASK embedding replaces selected-device context for uncommitted
  services.

Static service demand, compatibility, device capacity, dependency incidence,
and application structure already exist in the graph. The MVP does not add a
dense service-device-pair graph or a neural critic.

### 3.4 Shared Conditional Decoder

Both Stage 3 decoders start from all MASK and use the same model, feasibility
mask, commit schedule, verifier, repair, fallback, and exact final selection.

Before committing a service-device pair, the decoder masks choices that violate
compatibility, residual capacity, or direct-link feasibility with committed
dependency neighbors. If a service has no currently admissible choice, it
remains masked for a later step.

- **Deterministic conditional baseline:** commit the highest-confidence
  admissible choices with deterministic tie breaking.
- **Stochastic masked diffusion:** sample admissible choices for multiple
  particles from the same logits and schedule.

The commit counts per transition are identical between the two decoders. This
keeps the comparison focused on stochastic diffusion rather than on extra GNN
forwards or a different conditioning schedule.

## 4. Isolated Implementation Map

New modules are preferred over behavior-changing edits to frozen components:

- `diffusion/partial_mask.py`: partial-state validation, mask schedule,
  corruption, and deterministic replay;
- `graph/partial_context.py`: committed-load, residual-capacity, and visible
  dependency context;
- `models/conditional_denoiser.py`: typed conditional model with MASK state;
- `training/masked_objectives.py`: hidden-service reconstruction metrics;
- `training/masked_trainer.py`: independent `masked_conditional` checkpoint
  contract;
- `inference/masked_decode.py`: shared deterministic/stochastic completion;
- `experiments/phase6ee_stage3.py`: data locks, pilot gates, aggregation, and
  evidence freezes;
- dedicated configs and scripts numbered after the Stage 2B entrypoints.

Existing modules may receive small additive exports or generic helpers, but
old checkpoint parameter names, old inference defaults, and old experiment
artifacts must remain unchanged.

## 5. Development Data Contract

Create a new synthetic development dataset with a fresh master seed and the
same optimization semantics as Section II:

- 256 training instances;
- 64 checkpoint-selection instances;
- 64 untouched pilot-gate instances;
- service/device ranges aligned with the existing in-distribution medium
  regime;
- equivalent-MILP solution pools with target depth 16; a smaller pool is
  accepted only when no-good-cut enumeration proves that fewer unique feasible
  placements exist;
- dataset, pool, and split hashes frozen before training.

The checkpoint-selection and pilot-gate IDs must be disjoint. The pilot gate is
opened once after model/checkpoint/budget selection. A separate Stage 3 final
generation contract and master seed will be hash-locked now but its instances
will not be generated or evaluated until the multi-seed method is frozen.

Target supervision is fixed to best-only verified placements for the pilot,
because this was the strongest frozen Phase 6E-B target policy. The retrained
direct control uses the identical targets and training-step budget.

## 6. Staged Execution and Gates

### Stage 3.0: Contract and Regression Baseline

Status: complete. The deterministic contract lock is frozen, the legacy
regression passes 117 tests, and Stage 3 final data does not exist.

- Freeze this plan, old evidence hashes, new module boundaries, data seed, and
  forbidden partitions.
- Re-run the existing 99-test suite before model edits.
- Add no behavior-changing code in this stage.

Acceptance: deterministic plan/config hashes, 99 existing tests pass, and no
Stage 3 final data exists.

### Stage 3.1: State, Corruption, and Context MVP

Status: complete. Five focused state/context tests pass.

- Implement partial-state validation and monotone absorbing-MASK corruption.
- Implement partial dynamic context without changing `build_dynamic_context`.
- Add unit tests for padding, compatibility, `rho_0`, `rho_T`, monotonicity,
  deterministic replay, committed loads, residual capacity, and visible links.

Acceptance: CPU/CUDA tensor tests pass and legacy diffusion outputs remain
unchanged for fixed seeds.

### Stage 3.2: Conditional Model and Toy Overfit

Status: complete. The CUDA toy gate reaches 100% minimum hidden-service
accuracy and exact checkpoint replay.

- Implement the conditional denoiser and masked-only CE.
- Overfit 8--16 labeled toy instances across multiple mask ratios.
- Verify reconstruction from single-hidden, half-hidden, and all-hidden states.

Acceptance: at least 99% hidden-service accuracy on the toy set, exact
checkpoint resume, finite gradients, and no incompatible logits selected.

### Stage 3.3: Shared Decoders and Engineering Comparison

Status: complete. Deterministic/stochastic decoding, hard residual masks,
termination, and verified post-processing pass focused tests.

- Implement deterministic and stochastic completion from one model API.
- Apply hard residual masks before every commit.
- Reuse the existing proposal verifier, bounded repair, fallback, and exact
  evaluator.
- Profile candidate quality and runtime on toy/smoke data only.

Acceptance: deterministic replay, termination with no infinite loop, explicit
failure when budgets exhaust, and 100% verification for reported successes.

### Stage 3.4: New Development Dataset

Status: complete. All 384 instances are labeled and frozen; 6106 verified
placements are available under the best-only target contract.

- Generate, audit, label, and freeze the 384-instance Stage 3 development set.
- Lock one training seed, target policy, model dimensions, optimizer, step
  budget, commit schedule, and checkpoint rank before training.

Acceptance: zero split overlap, all solution pools verified by the exact
evaluator, and a reproducible dataset freeze.

### Stage 3.5: One-Seed Pilot

Status: complete and frozen with Outcome B. Direct `K=32` was selected on
checkpoint-selection, and the one-time 64-instance pilot was opened only after
the 121-test regression passed.

Train under matched data and step budgets:

1. one-shot direct control;
2. masked conditional model.

Evaluate on checkpoint-selection only to choose checkpoints and match online
budgets. Then open the 64-instance pilot gate once for:

1. direct one-shot;
2. deterministic conditional completion;
3. stochastic masked diffusion;
4. frozen categorical diffusion/rescue as historical anchors.

**Gate R3-B, partial conditioning:** proceed only if deterministic conditional
completion improves direct prediction by at least 10% relative verified/pre-
fallback gap or at least 5 percentage points raw-any feasibility, without
reducing final verified success. Pilot time may be at most 150% of direct.

**Gate R3-C, diffusion-specific value:** open multi-seed training only if the
stochastic decoder improves deterministic conditional completion by at least
5% relative gap, has more paired wins than losses, does not reduce raw-any
feasibility, and stays within 110% of its matched online time.

These are engineering continuation gates, not statistical-significance claims.

Observed Gate R3-B: pass (30.77% relative pre-fallback-gap improvement,
+46.875 percentage points raw feasibility, and 1.289x total time).

Observed Gate R3-C: fail on timing only (31.14% relative gap improvement,
40/8/16 paired wins/losses/ties, unchanged raw feasibility, but 1.620x total
time versus the 1.10 ceiling).

### Stage 3.6: Efficiency Confirmation Before Multi-Seed Freeze

Status: complete at Stage 3.6A and closed. No K passes the unchanged gate;
confirmation and diffusion-specific multi-seed work are not authorized.

- Do not reuse the consumed pilot for model, proposal-budget, or threshold
  selection.
- Profile stochastic K=2/K=4 and batched graph execution on checkpoint-
  selection only; preserve the frozen checkpoint and method semantics.
- Freeze one efficiency-oriented stochastic contract before generating a fresh
  independent confirmation split.
- Train three independent seeds only if the new confirmation satisfies the
  unchanged diffusion-specific quality and timing requirements.
- Freeze checkpoints, parameter counts, training time, and inference budgets.
- Generate/open the pre-registered Stage 3 final contract exactly once.
- Report mean/std, paired outcomes, raw feasibility, pre-fallback gap,
  repair/fallback burden, verified gap, and online time.

## 7. Decision Outcomes

- **Outcome A:** stochastic masked diffusion beats both direct and
  deterministic conditional decoding. Diffusion-specific manuscript claims are
  supportable.
- **Outcome B:** deterministic conditioning improves direct prediction, while
  stochastic decoding does not satisfy the complete diffusion-specific gate.
  Reframe the core as a typed graph conditional placement generator; retain
  stochastic decoding only as a quality-runtime option unless fresh evidence
  clears the efficiency gate.
- **Outcome C:** one-shot direct remains stronger. Stop diffusion redesign and
  remove solver-superiority claims.

The Stage 3.5--3.6 mechanism-level decision is Outcome B: stochastic decoding
does not beat deterministic conditional completion under the registered gate.
Stage 3.7 later passes a separate method-level matched-time gate against the
one-pass Direct baseline and therefore authorizes a newly sealed multi-seed
comparison. Sections IV--V remain unchanged until that confirmation is frozen;
the checkpoint-selection result is development evidence only.
