# Factor-Graph Categorical Diffusion Implementation

This directory contains the from-scratch implementation aligned with the
revised optimization model and typed factor-graph categorical diffusion method.

The project-local Python environment is located at `D:\GDM Paper\.venv`.
Use it directly without activating it:

```powershell
& 'D:\GDM Paper\.venv\Scripts\python.exe' --version
```

Small Phase 0 dependencies are recorded in `requirements-phase0.txt`. The
large ML stack is recorded in `requirements-ml.txt`; early installation notes
are archived under `..\history\2026-06-17_records_archive\implementation\`.
The resolved environment will be frozen to `requirements-lock.txt` after
installation and verification.

## Core Validation Commands

```powershell
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\00_check_environment.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\01_create_toy_instance.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\02_validate_ground_truth.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\03_generate_dataset.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\04_audit_dataset.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\05_stress_generator.py' --count 100

& 'D:\GDM Paper\.venv\Scripts\python.exe' -m pytest `
  'D:\GDM Paper\implementation\tests'
```

Gurobi-backed commands must run under the real Windows user associated with
the local academic license.

## Current Method Status

- Phases 0--3: deterministic ground truth, categorical diffusion, and typed
  factor denoiser complete.
- Phase 4A: smoke training pipeline complete.
- Phase 4B: constrained reverse solving, repair, fallback, and verification
  complete.
- Phase 4C small no-retraining validation complete.
- Phase 5A resumable production-training infrastructure complete.
- Phase 5B 136-instance labeled pilot dataset complete and frozen.
- Phase 5C three-seed pilot Go/No-Go complete: GO to scientific experiments.
- Phase 6A scientific experiment manifests, shared evaluation, aggregation,
  table/figure export, and reproducibility acceptance complete.
- Phase 6B dataset scaling study complete: final-data contracts selected after
  a frozen 56-instance probe through 48 services.
- Phase 6C final data and one-seed acceptance complete: GO to the fixed
  five-seed training campaign.
- Phase 6C five-seed training, aggregation, and checkpoint freeze complete.
- Phase 6D-A real time-limited MILP and independent greedy baseline adapters
  complete.
- Phase 6D-B direct categorical predictor implementation, validation
  acceptance, five-seed training, and checkpoint freeze complete.
- Phase 6D-C sealed main comparisons, final evidence freeze, and automatic
  paper-result report are complete.
- Phase 6E-A inference-only ablations and sensitivities are complete and
  frozen.
- Phase 6E-B target-sampling and soft-guidance retraining ablations are
  complete and frozen across three seeds and six variants.
- Phase 6E-D validation-locked time-matched diffusion/direct comparison is
  complete and frozen. Direct K=96 outperforms diffusion K=4 at nearly equal
  online time, so diffusion-superiority claims are not supported.
- Phase 6E-E Stage 1 is complete. Three-seed validation diagnostics select
  Stage 2 trajectory rescue.
- Phase 6E-E Stage 2A is complete and frozen. `rescue_all_five_b12` passes the
  untouched validation confirmation gate; the full regression passes 97 tests.

Important commands:

```powershell
# Audited smoke training
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\13_train_phase4a_smoke.py'

# Recommended small-scale constrained inference
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\15_validate_phase4b_inference.py' `
  --config 'D:\GDM Paper\implementation\configs\inference_phase4c_recommended_small.yaml'

# Reverse-step and hybrid diagnostic without retraining
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\16_validate_phase4c_small.py'

# Resumable production trainer; use --resume latest_checkpoint.pt after interruption
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\17_train_production.py' `
  --config 'D:\GDM Paper\implementation\configs\training_phase5c_pilot.yaml'

# Re-audit and freeze a completely labeled dataset
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\18_freeze_labeled_dataset.py' `
  'D:\GDM Paper\implementation\artifacts\datasets\phase5b-pilot'

# Aggregate completed multi-seed pilot evaluations
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\19_aggregate_phase5c_pilot.py' `
  'D:\GDM Paper\implementation\artifacts\phase5c-evaluations'

# Run one validated scientific experiment manifest
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\20_run_scientific_experiment.py' `
  'D:\GDM Paper\implementation\configs\experiment_phase6a_acceptance.yaml'

# Aggregate compatible scientific experiment runs
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\21_aggregate_scientific_experiments.py' `
  <run-directory-1> <run-directory-2> --output <aggregate.json>

# Export standard paper-ready figures from raw experiment artifacts
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\22_export_experiment_figures.py' `
  <run-directory>

# Profile a frozen labeled scale-probe dataset
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\23_profile_scale_probe.py' `
  <dataset-root> <checkpoint>

# Probe deeper solution pools on selected hard instances
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\24_probe_pool_depth.py' `
  <dataset-root> <instance-id> --target-size 16 --time-limit 120 `
  --output <output.json>

# Export scale-probe diagnostic figures
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\25_export_scale_probe_figures.py' `
  <scale-probe-run-directory>

# Resume labeling using an immutable final-data contract
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\26_label_contract_dataset.py' `
  <dataset-root>

# Export one-run training diagnostics
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\27_export_training_curves.py' `
  <training-run-directory>

# Aggregate and freeze the five independent final-training seeds
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\28_aggregate_phase6c_training.py' `
  <run-directory-1> <run-directory-2> <run-directory-3> `
  <run-directory-4> <run-directory-5>

# Verify the frozen final-training checkpoints before evaluation
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\29_verify_phase6c_checkpoint_freeze.py' `
  'D:\GDM Paper\implementation\artifacts\phase6c-five-seed\checkpoint_freeze.json'

# Re-run the non-final-data Phase 6D baseline-adapter acceptance
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\20_run_scientific_experiment.py' `
  'D:\GDM Paper\implementation\configs\experiment_phase6d_baseline_acceptance.yaml'

# Verify and regenerate the frozen Phase 6D-C aggregates
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\33_finalize_phase6d_c_campaign.py'

# Generate paper tables, figures, and the reviewer-aware Phase 6D-C report
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\34_summarize_phase6d_c_results.py'

# Regenerate and verify the locked Phase 6E-A inference-ablation manifests
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\35_prepare_phase6e_a_campaign.py'

# Run one or more resumable Phase 6E-A groups
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\36_run_phase6e_a_campaign.py' `
  --group postprocessing

# Aggregate and freeze Phase 6E-A after all four groups complete
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\37_finalize_phase6e_a_campaign.py'

# Regenerate and verify the deterministic Phase 6E-B training campaign
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\38_prepare_phase6e_b_campaign.py'

# Idempotently run or resume Phase 6E-B training
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\39_run_phase6e_b_training.py' `
  --mode full

# Freeze and verify all six three-seed checkpoint groups
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\40_finalize_phase6e_b_training.py'

# Prepare, run, and freeze paired final-ID mechanism evaluation
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\41_prepare_phase6e_b_evaluation.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\42_run_phase6e_b_evaluation.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\43_finalize_phase6e_b_evaluation.py'

# Reproduce the validation-only Phase 6E-E Stage 1 diagnostic freeze
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\50_prepare_phase6e_e_stage1.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\51_run_phase6e_e_stage1.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\52_finalize_phase6e_e_stage1.py'

# Reproduce the validation-only Stage 2A calibration and confirmation freeze
& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\53_prepare_phase6e_e_stage2a.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\54_run_phase6e_e_stage2a_calibration.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\55_finalize_phase6e_e_stage2a_calibration.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\56_run_phase6e_e_stage2a_confirmation.py'

& 'D:\GDM Paper\.venv\Scripts\python.exe' `
  'D:\GDM Paper\implementation\scripts\57_finalize_phase6e_e_stage2a_confirmation.py'
```

Do not use the smoke training configuration for final training. The historical
production-training readiness audit is archived under
`..\history\2026-06-17_records_archive\implementation\`.

Install the local package after recreating the virtual environment:

```powershell
& 'D:\GDM Paper\.venv\Scripts\python.exe' -m pip install `
  --editable 'D:\GDM Paper\implementation' --no-deps --no-build-isolation
```

See `SCHEMA.md` for the shared instance-data contract.
Archived Phase 0--5 planning/status files are under
`..\history\2026-06-17_records_archive\implementation\`.
See `DATASET_DESIGN.md` for Phase 1B generation and partition conventions.
See `PHASE6_STATUS.md` for scientific-experiment infrastructure and evidence
boundaries.
See `PHASE6B_SCALE_PROBE_REPORT.md` for the final-data scaling decision.
See `PHASE6C_STATUS.md` for frozen final-data and one-seed training acceptance.
See `PHASE6D_PLAN.md` for strong-baseline contracts and sealed-test gating.
See `PHASE6D_B_STATUS.md` for direct-predictor acceptance and freeze evidence.
See `PHASE6D_C_PROTOCOL.md` and `PHASE6D_C_RESULTS.md` for the sealed final
comparison contract, results, and evidence boundaries.
See `PHASE6E_PLAN.md`, `PHASE6E_A_STATUS.md`, `PHASE6E_B_PLAN.md`, and
`PHASE6E_B_STATUS.md` for the completed inference-only and retraining-based
ablation campaigns. See `PHASE6E_B_REPORT_ZH.md` for the concise Chinese
interpretation and manuscript claim boundary.
See `PHASE6E_D_PLAN.md`, `PHASE6E_D_STATUS.md`, and `PHASE6E_D_REPORT_ZH.md`
for the matched-time adverse result. See `PHASE6E_E_PLAN.md` and
`PHASE6E_E_STATUS.md` for the active diagnosis-first improvement phase, and
`PHASE6E_E_STAGE1_REPORT_ZH.md` plus `PHASE6E_E_STAGE2A_REPORT_ZH.md` for the
frozen diagnostic and trajectory-rescue interpretations.
See `..\current_handoff_2026-06-21_diffusion_improvement_plan.md` for the
current authoritative handoff. Earlier handoffs are under `..\history\`.
See `..\experiment_engineering_checklist_zh.md` for the current paper-level
experiment and evidence gates.
