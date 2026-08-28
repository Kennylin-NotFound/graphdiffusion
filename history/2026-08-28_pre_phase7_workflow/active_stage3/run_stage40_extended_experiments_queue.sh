#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/GDM_Paper/implementation}"
OUT_REL="${OUT_REL:-artifacts/phase6e-e-stage40-extended-experiments}"
OUT="$ROOT/$OUT_REL"

mkdir -p "$OUT/logs"
STATUS="$OUT/queue_status.tsv"

{
  echo -e "event\tutc\tstage\tpid\tstatus"
  echo -e "start\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\tall\t$$\trunning"
} > "$STATUS"

run_stage() {
  local stage="$1"
  local script="$2"
  local log="$OUT/logs/${stage}.log"
  echo -e "stage_start\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t${stage}\t$$\trunning" >> "$STATUS"
  (
    cd "$ROOT"
    bash "$script"
  ) > "$log" 2>&1
  echo -e "stage_finish\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t${stage}\t$$\tok" >> "$STATUS"
}

run_stage "cross_scale_strong_heuristics" "$ROOT/active_stage3/run_stage40_cross_scale_strong_heuristics.sh"
run_stage "controlled_shift_10seed" "$ROOT/active_stage3/run_stage39_controlled_shift_queue.sh"
run_stage "realistic_profile_10seed" "$ROOT/active_stage3/run_stage40_realistic_profile_queue.sh"

echo -e "finish\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\tall\t$$\tok" >> "$STATUS"
