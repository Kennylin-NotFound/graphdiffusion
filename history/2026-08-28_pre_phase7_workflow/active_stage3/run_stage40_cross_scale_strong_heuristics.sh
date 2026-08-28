#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/GDM_Paper/implementation}"
PY="${PY:-/home/linchen/miniconda3/envs/gdm-stage39/bin/python}"
OUT_REL="${OUT_REL:-artifacts/phase6e-e-cross-scale-strong-heuristics}"
SCRIPT="$ROOT/scripts/81_run_cross_scale_strong_heuristics.py"

mkdir -p "$ROOT/$OUT_REL/logs"
STATUS="$ROOT/$OUT_REL/queue_status.tsv"

{
  echo -e "event\tutc\tpid\tstatus"
  echo -e "start\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t$$\trunning"
} > "$STATUS"

(
  cd "$ROOT"
  "$PY" "$SCRIPT" \
    --dataset-root artifacts/datasets/phase6c-final-scale \
    --output-root "$OUT_REL"
) > "$ROOT/$OUT_REL/logs/run.log" 2>&1

echo -e "finish\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t$$\tok" >> "$STATUS"
