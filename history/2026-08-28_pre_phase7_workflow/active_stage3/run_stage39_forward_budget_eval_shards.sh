#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/GDM_Paper/implementation}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/gdm-stage39/bin/python}"
OUTPUT="$ROOT/artifacts/phase6e-e-stage39-forward-budget-evaluation"
LOG_DIR="$OUTPUT/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT"

shards=(
  "0:2026070111,2026070112,2026070113"
  "1:2026070114,2026070115,2026070116"
  "2:2026070117,2026070118"
  "3:2026070119,2026070120"
)

for shard in "${shards[@]}"; do
  gpu="${shard%%:*}"
  seeds="${shard#*:}"
  log="$LOG_DIR/forward_budget_gpu${gpu}.log"
  if pgrep -af "74_run_phase6e_e_stage39_forward_budget.py run --seeds $seeds" >/dev/null; then
    echo "already_running gpu=$gpu seeds=$seeds"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" \
    scripts/74_run_phase6e_e_stage39_forward_budget.py run --seeds "$seeds" \
    > "$log" 2>&1 &
  echo "started gpu=$gpu seeds=$seeds pid=$! log=$log"
done
