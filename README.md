# Phase 7 Implementation

This directory contains the implementation and frozen evidence aligned with the
current proposal-conditioned-recovery manuscript. Earlier Stage 3--6F campaigns
remain available under `history/` and `artifacts/history/`, but they are not
current experiment entrypoints.

## Current Evidence

Read `artifacts/CURRENT_EVIDENCE.md` before opening individual experiment
directories. The primary manuscript evidence is:

- `artifacts/paper-evidence-phase7-advisor-revision/`
- `artifacts/phase7-proposal-conditioned-recovery/`
- `artifacts/phase7-proposal-conditioned-recovery-controlled/`
- `artifacts/phase7-proposal-conditioned-recovery-realistic/`

The retained Phase 6E/6F directories at the artifact root provide checkpoints,
dataset references, training curves, or forward-budget calibration required by
the Phase 7 evidence pipeline. They are dependencies, not competing current
method versions.

The `runs/`, `phase6c-five-seed/`, `phase6d-c-final/`,
`phase6d-direct-five-seed/`, `phase6e-a-inference/`, `phase6e-b-training/`,
`phase6e-b-evaluation/`, `phase6e-e-stage2b/`, and `phase6e-e-stage3/`
directories are legacy regression fixtures. `PHASE6E_E_STAGE3_PLAN.md` is
likewise retained at its original path because its content hash belongs to the
Stage 3 contract. These paths are intentionally audited by the regression
suite.

## Current Entry Points

- `scripts/97_run_phase7_proposal_recovery_evaluation.py`: Phase 7 evaluation.
- `scripts/98_export_phase7_manuscript_evidence.py`: unified evidence export.
- `active_phase7/`: guarded PowerShell resume/orchestration scripts retained for
  reproducibility; no background job is implied by the directory name.
- `src/gdm_factor_diffusion/`: active solver, graph, model, training, and
  inference implementation.
- `tests/`: regression tests for optimization semantics and evaluation logic.

## Validation

```powershell
& 'D:\GDM Paper\.venv\Scripts\python.exe' -m pytest `
  'D:\GDM Paper\implementation\tests' -q

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\00_check_environment.py'
```

Gurobi-backed checks must run under the Windows account associated with the
local license.

## Layout

- `configs/`: frozen dataset and training configurations needed for provenance.
- `scripts/`: data, training, evaluation, and evidence-export scripts. Numbered
  scripts are retained because current Phase 7 evaluation imports frozen helper
  logic from Script 92.
- `artifacts/`: current evidence, its direct dependencies, and the explicitly
  documented regression fixtures whose frozen paths are part of the test
  contract.
- `artifacts/history/2026-08-28_pre_phase7_archive/`: earlier experiments,
  smoke runs, superseded policies, and transfer archives.
- `history/2026-08-28_pre_phase7_workflow/`: Stage 3 launchers and reports.

The reorganization does not change source code, model checkpoints, datasets,
or manuscript evidence values.
