# Pre-Stage-3 Entry Point Archive

This directory contains scripts and configurations used by Phases 0--6E-D and
the Stage 1/2 diagnostics. They are retained for evidence provenance and old-run
reproduction, but they are not valid launchers for the absorbing-MASK Stage 3
training contract.

The current entrypoint is `implementation/active_stage3/run_training.ps1`.
The former command-heavy implementation README is retained as
`README_pre_stage3.md` for historical reproduction only.
Frozen datasets, checkpoints, and evidence remain in their original artifact
locations and were not moved.

Archive policy:

- `scripts/`: old phase-specific launchers and diagnostics;
- `configs/`: old training, inference, campaign, and generated configurations;
- active Stage 3 files and general dataset/audit utilities remain in their
  original `implementation/scripts` and `implementation/configs` locations;
- tests remain active because the complete suite is fast and protects both old
  optimization semantics and the new training contract.

Archived inventory: 55 Python scripts and 105 YAML configurations. Historical
config readers use a missing-path-only archive fallback and still verify the
original frozen SHA-256 values.
