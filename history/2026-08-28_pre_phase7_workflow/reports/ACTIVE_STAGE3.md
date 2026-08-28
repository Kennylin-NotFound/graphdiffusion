# Active Stage 3 Implementation

The only supported formal-training entrypoint is:

`active_stage3/run_training.ps1`

It verifies the frozen pre-training contract before dispatching to
`scripts/64_train_phase6e_e_stage3.py` with the locked Stage 3 configuration.
Do not use archived Phase 4--6E-D training scripts for the redesigned model.

Active documentation:

- `active_stage3/METHOD_REWRITE_REFERENCE.md`: method and notation contract for
  later manuscript revision;
- `active_stage3/README.md`: authoritative files, launch commands, and archive
  policy;
- `PHASE6E_E_STAGE3_PRETRAINING_REPORT_ZH.md`: verified engineering evidence;
- `PHASE6E_E_STAGE3_PLAN.md`: experiment gates and claim boundary.

Historical scripts/configurations are retained under
`history/pre_stage3_2026-06-21/` for reproducibility, not execution.
