# Phase 6C Final Data And Training Status

Updated: 2026-06-15

## Current Verdict

Phase 6C is complete. Final-data preparation, one-seed acceptance, five-seed
training, cross-seed aggregation, and cryptographic checkpoint freezing are
complete. Final test-set comparisons, baselines, ablations, and Section V
claims remain pending.

## Frozen Final Data

Two separately frozen contracts prevent training/test leakage.

### Main Training And Generalization Data

- root: `artifacts/datasets/phase6c-final-main`;
- instances: 1,024;
- partitions: 512 train, 64 validation, 128 ID test, and five 64-instance
  controlled shifts;
- target pool size: 16;
- verified stored placements: 16,047;
- 990 instances reached 16 solutions;
- 34 instances exhausted their complete feasible space before 16 solutions;
- total recorded Gurobi pool time: approximately 290.73 seconds.

Every shift is machine-checked against the training distribution. Validation
and ID testing retain the same distribution fields; sparse topology, low
compatibility, high sharing, and unseen workflow change only their approved
factors. The unseen-size partition remains within the 19--24-service upper
seen-medium regime.

### Test-Only Scalability Data

- root: `artifacts/datasets/phase6c-final-scale`;
- instances: 112;
- partitions: 64 medium, 32 large, and 16 extra-large;
- service range: 9--48;
- target pool size: 4;
- verified stored placements: 448;
- every pool reached its target;
- total recorded Gurobi pool time: approximately 32.77 seconds.

Both datasets passed instance, witness, graph-relation, pool-checksum,
hard-feasibility, exact-latency, energy-distribution, and freeze-hash audits.

## Reproducibility Hardening

- final dataset configurations include machine-checked count, role,
  controlled-shift, and labeling contracts;
- `26_label_contract_dataset.py` reads the immutable labeling contract instead
  of accepting ad hoc pool settings;
- freezing rejects a solution-pool configuration that disagrees with the
  dataset contract;
- production training can require a frozen dataset and records the freeze and
  core-manifest hashes;
- CUDA RNG restoration now handles checkpoints mapped to CUDA correctly;
- training diagnostics are exported only as explicitly labeled single-seed
  evidence.

## One-Seed Acceptance Training

Training run:
`artifacts/runs/20260615T070248Z-phase6c-final-acceptance`

Configuration:

- 512 training and 64 validation instances;
- batch size 32;
- hidden dimension 128 and four typed factor-message layers;
- energy-weighted targets with capacity/link guidance;
- 25 reverse transitions and four proposals for constrained validation;
- fixed first 16 validation instances for checkpoint selection;
- 1,500 training steps after a verified resume from the initial 1,000-step
  budget.

The best checkpoint occurs at step 1,300:

- constrained-validation verified rate: 100%;
- mean gap to pool best: approximately 0.487%;
- raw feasible rate: approximately 35.94%;
- wins/ties/losses versus fallback: 7/9/0 on the fixed 16-instance selection
  subset.

The gap improves through step 1,300 and then rises slightly at steps 1,400 and
1,500. The selected fixed budget for the five-seed campaign is therefore 1,500
steps with constrained-validation checkpoint selection.

## Full-Validation Acceptance Evaluation

Evaluation run:
`artifacts/phase6c-acceptance/20260615T071345Z-phase6c-acceptance`

On all 64 validation instances:

| Method | Mean gap to pool best | Raw feasible rate | Mean online time |
|---|---:|---:|---:|
| Learned Hybrid | 1.080% | 32.812% | 0.641 s |
| Random Hybrid | 2.386% | 1.172% | 0.0276 s |
| Fallback Only | 2.401% | not applicable | 0.00370 s |

The learned hybrid improves fallback on 22 instances, ties on 42, and loses on
none because fallback remains available during final selection. Relative to
fallback, its mean validation gap is reduced by approximately 55.01%. All 192
evaluated method outputs are feasible, and every pool-best reference is traced
to a proven-optimal first MILP solve.

This is validation evidence used only for the training Go/No-Go decision. It
is not a final paper result.

## Acceptance Gate

Status: GO

The five-seed campaign used `configs/training_phase6c_final.yaml`, changing
only the seed and run name.

## Five-Seed Final Training

Frozen campaign:
`artifacts/phase6c-five-seed/checkpoint_freeze.json`

Seeds: 20260622--20260626.

All five runs:

- use the identical frozen main dataset and training contract;
- complete the fixed 1,500-step budget;
- select checkpoints only on the same fixed 16-instance validation subset;
- achieve 100% verified rate at the selected checkpoint;
- remain independent seeds rather than repeated executions.

| Seed | Best step | Best validation gap | Raw feasible rate | Wins over fallback |
|---:|---:|---:|---:|---:|
| 20260622 | 1300 | 0.487% | 35.94% | 7 |
| 20260623 | 1000 | 0.541% | 35.94% | 6 |
| 20260624 | 1300 | 0.457% | 23.44% | 6 |
| 20260625 | 500 | 0.600% | 32.81% | 4 |
| 20260626 | 1500 | 0.530% | 34.38% | 6 |

Across selected checkpoints:

- mean validation gap: approximately 0.523%, with 0.055% cross-seed standard
  deviation;
- mean raw feasible rate: 32.50%, with 5.23% cross-seed standard deviation;
- mean wins over fallback: 5.8 of 16 validation instances;
- selected checkpoint step: 500--1500, mean 1120.

The dispersed selected steps and non-monotonic constrained gap confirm that
last-step selection or denoising-loss-only selection would be methodologically
incorrect.

The checkpoint freeze records SHA-256 hashes for every selected model,
training log, and run summary. It is independently verifiable with
`scripts/29_verify_phase6c_checkpoint_freeze.py`.

## Phase 6D Entry

Status: Phase 6D-A and Phase 6D-B complete; Phase 6D-C next

Phase 6D may evaluate only the five frozen diffusion and five frozen direct
predictor checkpoints. ID, controlled-shift, and scale test partitions remained
unused during checkpoint selection. Time-limited MILP, independent greedy, and
the direct categorical predictor have passed acceptance. Phase 6D-C must fix
final manifests and reporting rules before opening sealed partitions.
