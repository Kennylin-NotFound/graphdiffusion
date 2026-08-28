# Phase 6E-E Masked Diffusion Status

Updated: 2026-06-23

## Active Status

The active method is the absorbing-MASK partial-assignment graph diffusion
solver. Old final-state-only diffusion diagnostics, trajectory rescue,
matched-time comparisons, and training-mechanism ablations have been archived
under:

`history/2026-06-23_old_diffusion_experiment_archive/`

Those archived reports are provenance only. They should not be used as active
Section V evidence for the new method.

## Active Evidence

- Stage 3.8 training is complete and frozen for three seeds:
  `2026070111`, `2026070112`, and `2026070113`.
- Training freeze:
  `implementation/artifacts/phase6e-e-stage38-training/training_freeze.json`.
- Training freeze SHA-256:
  `C9C9F1799208B5F75B53C73ACFE55D587B68DC2DAA0254026AC7FCBF162FA4F0`.
- Stage 3.8 sealed ID dataset contains 128 fresh instances and 2048 verified
  MILP solution-pool placements.
- Stage 3.8 sealed evaluation is complete for Direct K=64, Masked
  deterministic K=1, and Masked Diffusion K=8.
- Final evidence:
  `implementation/artifacts/phase6e-e-stage38-sealed-evaluation/final_evidence.json`.
- Final evidence SHA-256:
  `DDD1E46EEA36B45A333C6DEB26148F5C2A74FB2D5A03FF8C8598D0201DB0D636`.
- Decision-lock SHA-256:
  `9C9CC6A20F71A4D8F9EB93EFBD8490C141119345684B5FA0A1085E5F5ED9EF96`.
- Raw-only supplemental evaluation is complete with repair and fallback
  disabled:
  `implementation/artifacts/phase6e-e-stage38-rawonly-evaluation/rawonly_evidence.json`.
- Latest full implementation regression after Stage 3.8 finalization:
  `135 passed`.

## Current Result Boundary

Masked Diffusion K=8 has strong raw neural proposal evidence:

- raw success: 95.57% versus 48.96% for Direct K=64;
- raw gap: 2.985% versus 4.009% for Direct K=64;
- raw paired comparison: 110 wins, 12 losses, 6 ties, p = 5.47e-21.

The full verified pipeline shows a favorable aggregate trend:

- final gap: 1.254% for Masked Diffusion K=8 versus 1.631% for Direct K=64;
- pre-fallback gap: 2.912% versus 3.693%;
- total time: 0.486 s versus 0.501 s.

However, the strict pre-registered Stage 3.8 gate does not pass:

- `diffusion_claim_confirmed=false`;
- seed `2026070112` is adverse in pre-fallback/final gap;
- instance-level pre-fallback sign test is p = 0.101.

## Active Claim Boundary

Safe:

- absorbing-MASK diffusion substantially improves raw neural proposal
  feasibility and raw candidate quality over the one-pass Direct K=64 baseline;
- hard verification, bounded repair, and deterministic fallback remain part of
  the final feasibility pipeline;
- the full pipeline shows favorable aggregate quality at comparable online
  time.

Unsafe:

- do not claim statistically confirmed end-to-end dominance over Direct K=64;
- do not reuse old final-state-only diffusion tables as new masked-diffusion
  evidence;
- do not claim energy weighting or soft guidance is necessary unless new
  masked-specific ablations are run;
- do not claim robustness, scalability, or realistic-trace validation until
  masked-specific experiments support those claims.

## Active Planning File

Use this file for the next experiment tasks:

`masked_diffusion_experiment_checklist_zh.md`

Recommended next stage:

1. Run or assemble the new masked baseline suite on the Stage 3.8 sealed ID set.
2. Complete the masked repair-only post-processing ablation.
3. Decide whether to run proposal-budget sensitivity and controlled-shift tests
   before rewriting Section V.
