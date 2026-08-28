#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/GDM_Paper/implementation}"
PY="${PY:-/home/linchen/miniconda3/envs/gdm-stage39/bin/python}"
OUT_REL="${OUT_REL:-artifacts/phase6e-e-cross-scale-evaluation-10seed}"
OUT="$ROOT/$OUT_REL"
SCRIPT="$ROOT/scripts/76_run_phase6e_e_cross_scale_evaluation.py"
METHOD_PROFILE="${METHOD_PROFILE:-core}"
REPAIR_CANDIDATE_LIMIT="${REPAIR_CANDIDATE_LIMIT:-16}"
REPAIR_MAX_MOVES="${REPAIR_MAX_MOVES:-8}"
FALLBACK_MAX_SEARCH_NODES="${FALLBACK_MAX_SEARCH_NODES:-30000}"

mkdir -p "$OUT/logs"
STATUS="$OUT/queue_status.tsv"

{
  echo -e "event\tutc\tgpu\tseeds\tpid\tstatus"
} > "$STATUS"

SEED_SHARDS=(
  "2026070111,2026070112,2026070113"
  "2026070114,2026070115,2026070116"
  "2026070117,2026070118"
  "2026070119,2026070120"
)

pids=()
for index in "${!SEED_SHARDS[@]}"; do
  seeds="${SEED_SHARDS[$index]}"
  log="$OUT/logs/cross_scale_gpu${index}.log"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$index" "$PY" "$SCRIPT" run \
      --seeds "$seeds" \
      --output-root "$OUT_REL" \
      --device cuda \
      --method-profile "$METHOD_PROFILE" \
      --repair-candidate-limit "$REPAIR_CANDIDATE_LIMIT" \
      --repair-max-moves "$REPAIR_MAX_MOVES" \
      --fallback-max-search-nodes "$FALLBACK_MAX_SEARCH_NODES"
  ) > "$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo -e "start\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t$index\t$seeds\t$pid\trunning" >> "$STATUS"
done

failed=0
for shard_index in "${!pids[@]}"; do
  pid="${pids[$shard_index]}"
  seeds="${SEED_SHARDS[$shard_index]}"
  if wait "$pid"; then
    echo -e "finish\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t$shard_index\t$seeds\t$pid\tok" >> "$STATUS"
  else
    rc=$?
    echo -e "finish\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t$shard_index\t$seeds\t$pid\tfailed:$rc" >> "$STATUS"
    failed=1
  fi
done

if [[ "$failed" -eq 0 ]]; then
  (
    cd "$ROOT"
    "$PY" "$SCRIPT" finalize \
      --seeds "2026070111,2026070112,2026070113,2026070114,2026070115,2026070116,2026070117,2026070118,2026070119,2026070120" \
      --output-root "$OUT_REL" \
      --device cuda \
      --method-profile "$METHOD_PROFILE" \
      --repair-candidate-limit "$REPAIR_CANDIDATE_LIMIT" \
      --repair-max-moves "$REPAIR_MAX_MOVES" \
      --fallback-max-search-nodes "$FALLBACK_MAX_SEARCH_NODES"
  ) > "$OUT/logs/finalize.log" 2>&1
  echo -e "finalize\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\tall\tall\t$$\tok" >> "$STATUS"
else
  echo -e "finalize\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\tall\tall\t$$\tskipped_due_to_failed_shard" >> "$STATUS"
  exit 1
fi
