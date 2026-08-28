# Stage 3 Active Workspace

Updated: 2026-06-22

## Authoritative Entry

Use `run_training.ps1` for every formal Stage 3 training run. The launcher:

1. resolves the project-local `.venv`;
2. replays `scripts/65_finalize_phase6e_e_stage3_pretraining.py` and refuses to
   continue if source, configuration, data, or evidence hashes drift;
3. explicitly passes `configs/training_phase6e_e_stage3_pilot.yaml`;
4. dispatches only `direct` or `masked_conditional` to
   `scripts/64_train_phase6e_e_stage3.py`.

The internal Python trainer rejects direct invocation unless this launcher has
set its process-scoped authorization marker. This prevents an old command from
silently bypassing the locked config and freeze check.

Commands from `D:\GDM Paper\implementation`:

```powershell
.\active_stage3\run_training.ps1 -ModelKind direct -Preflight
.\active_stage3\run_training.ps1 -ModelKind masked_conditional -Preflight
.\active_stage3\run_training.ps1 -ModelKind direct
.\active_stage3\run_training.ps1 -ModelKind masked_conditional
```

Use `-Resume <checkpoint>` only with the matching model family.

## Active Files

Core Stage 3 implementation:

- `src/gdm_factor_diffusion/diffusion/partial_mask.py`
- `src/gdm_factor_diffusion/graph/partial_context.py`
- `src/gdm_factor_diffusion/models/conditional_denoiser.py`
- `src/gdm_factor_diffusion/training/masked_objectives.py`
- `src/gdm_factor_diffusion/training/masked_trainer.py`
- `src/gdm_factor_diffusion/training/stage3_production.py`
- `src/gdm_factor_diffusion/inference/masked_decode.py`
- `src/gdm_factor_diffusion/inference/masked_decode_vectorized.py`
- `src/gdm_factor_diffusion/experiments/phase6ee_stage37.py`

Locked configurations:

- `configs/phase6e_e_stage3_contract.yaml`
- `configs/dataset_phase6e_e_stage3_development.yaml`
- `configs/training_phase6e_e_stage3_pilot.yaml`
- `configs/phase6e_e_stage3_pilot.yaml`
- `configs/phase6e_e_stage36_efficiency.yaml`
- `configs/phase6e_e_stage37_matched_time.yaml`
- `configs/dataset_phase6e_e_stage36_confirmation.yaml` (preregistered but
  not generated)

Stage 3 scripts:

- `scripts/62_prepare_phase6e_e_stage3.py`
- `scripts/63_validate_phase6e_e_stage3_mvp.py`
- `scripts/64_train_phase6e_e_stage3.py`
- `scripts/65_finalize_phase6e_e_stage3_pretraining.py`
- `scripts/66_finalize_phase6e_e_stage3_training.py`
- `scripts/67_prepare_phase6e_e_stage3_pilot.py`
- `scripts/68_calibrate_phase6e_e_stage3_pilot.py`
- `scripts/69_run_phase6e_e_stage3_pilot.py`
- `scripts/70_run_phase6e_e_stage36_efficiency.py`
- `scripts/71_run_phase6e_e_stage37_matched_time.py`

General retained utilities:

- `scripts/00_check_environment.py`
- `scripts/03_generate_dataset.py`
- `scripts/04_audit_dataset.py`
- `scripts/07_generate_solution_pools.py`
- `scripts/08_audit_solution_pools.py`
- `scripts/18_freeze_labeled_dataset.py`
- `scripts/26_label_contract_dataset.py`

## Evidence Boundary

- Development data: `artifacts/datasets/phase6e-e-stage3-development/`
- Training-ready freeze: `artifacts/phase6e-e-stage3/pretraining_freeze.json`
- Completed one-seed training freeze:
  `artifacts/phase6e-e-stage3-training/training_freeze.json`
- Formal output root: `artifacts/phase6e-e-stage3-training/`
- One-time pilot evidence: `artifacts/phase6e-e-stage3-pilot/`
- Efficiency evidence: `artifacts/phase6e-e-stage36-calibration/`
- Method-level matched-time development evidence:
  `artifacts/phase6e-e-stage37-calibration/`
- Stage 3 final data and the newly authorized sealed multi-seed data remain
  unopened.

All tests remain active. The 131-test suite costs about ten seconds and
protects the frozen optimization semantics, old evidence readers, and new
Stage 3 behavior. Generated Python/pytest caches may be deleted safely.

The one-time pilot is consumed. Do not invoke
`69_run_phase6e_e_stage3_pilot.py run` again or use `pilot_gate` for
model/budget selection. See
`../PHASE6E_E_STAGE3_PILOT_REPORT_ZH.md` for the frozen Outcome B boundary.
Stage 3.6 selects no stochastic K, so its preregistered confirmation dataset
must remain absent. See `../PHASE6E_E_STAGE36_REPORT_ZH.md`.
Stage 3.7 separately passes the method-level matched-time development gate and
authorizes a new sealed multi-seed contract. See
`../PHASE6E_E_STAGE37_REPORT_ZH.md`; do not treat checkpoint-selection results
as final manuscript evidence.
