#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/GDM_Paper/implementation}"
PY="${PY:-/home/linchen/miniconda3/envs/gdm-stage39/bin/python}"
DATASET_REL="${DATASET_REL:-artifacts/datasets/phase6e-e-controlled-shift}"
CONFIG_REL="${CONFIG_REL:-configs/dataset_phase6e_e_controlled_shift.yaml}"
OUT_REL="${OUT_REL:-artifacts/phase6e-e-controlled-shift-evaluation-10seed}"
PARTITIONS="${PARTITIONS:-test_id_reference,shift_high_sharing,shift_low_compatibility,shift_tight_capacity,shift_unseen_workflow}"
MAX_PER_PARTITION="${MAX_PER_PARTITION:-}"
PREPARE_DATA="${PREPARE_DATA:-1}"
METHOD_PROFILE="${METHOD_PROFILE:-core}"
REPAIR_CANDIDATE_LIMIT="${REPAIR_CANDIDATE_LIMIT:-16}"
REPAIR_MAX_MOVES="${REPAIR_MAX_MOVES:-8}"
FALLBACK_MAX_SEARCH_NODES="${FALLBACK_MAX_SEARCH_NODES:-30000}"

DATASET="$ROOT/$DATASET_REL"
OUT="$ROOT/$OUT_REL"
SCRIPT="$ROOT/scripts/78_run_phase6e_e_controlled_shift_evaluation.py"

mkdir -p "$OUT/logs"
STATUS="$OUT/queue_status.tsv"

{
  echo -e "event\tutc\tgpu\tseeds\tpid\tstatus"
} > "$STATUS"

eval_extra=()
if [[ -n "$MAX_PER_PARTITION" ]]; then
  eval_extra+=(--max-instances-per-partition "$MAX_PER_PARTITION")
fi

if [[ "$PREPARE_DATA" == "1" ]]; then
  (
    cd "$ROOT"
    if [[ ! -f "$DATASET/manifest.json" ]]; then
      "$PY" "$ROOT/scripts/03_generate_dataset.py" --config "$ROOT/$CONFIG_REL"
    fi
    if [[ ! -f "$DATASET/solution_pool_manifest.json" ]]; then
      "$PY" "$ROOT/scripts/26_label_contract_dataset.py" "$DATASET"
    else
      "$PY" "$ROOT/scripts/26_label_contract_dataset.py" "$DATASET"
    fi
    if [[ ! -f "$DATASET/dataset_freeze.json" ]]; then
      "$PY" "$ROOT/scripts/18_freeze_labeled_dataset.py" "$DATASET"
    fi
  ) > "$OUT/logs/prepare_data.log" 2>&1
  echo -e "prepare_data\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\tall\tall\t$$\tok" >> "$STATUS"
fi

SEED_SHARDS=(
  "2026070111,2026070112,2026070113"
  "2026070114,2026070115,2026070116"
  "2026070117,2026070118"
  "2026070119,2026070120"
)

pids=()
for index in "${!SEED_SHARDS[@]}"; do
  seeds="${SEED_SHARDS[$index]}"
  log="$OUT/logs/controlled_shift_gpu${index}.log"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$index" "$PY" "$SCRIPT" run \
      --dataset-root "$DATASET_REL" \
      --seeds "$seeds" \
      --partitions "$PARTITIONS" \
      --output-root "$OUT_REL" \
      --device cuda \
      --method-profile "$METHOD_PROFILE" \
      --repair-candidate-limit "$REPAIR_CANDIDATE_LIMIT" \
      --repair-max-moves "$REPAIR_MAX_MOVES" \
      --fallback-max-search-nodes "$FALLBACK_MAX_SEARCH_NODES" \
      "${eval_extra[@]}"
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
      --dataset-root "$DATASET_REL" \
      --seeds "2026070111,2026070112,2026070113,2026070114,2026070115,2026070116,2026070117,2026070118,2026070119,2026070120" \
      --partitions "$PARTITIONS" \
      --output-root "$OUT_REL" \
      --device cuda \
      --method-profile "$METHOD_PROFILE" \
      --repair-candidate-limit "$REPAIR_CANDIDATE_LIMIT" \
      --repair-max-moves "$REPAIR_MAX_MOVES" \
      --fallback-max-search-nodes "$FALLBACK_MAX_SEARCH_NODES" \
      "${eval_extra[@]}"
  ) > "$OUT/logs/finalize.log" 2>&1
  echo -e "finalize\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\tall\tall\t$$\tok" >> "$STATUS"
else
  echo -e "finalize\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\tall\tall\t$$\tskipped_due_to_failed_shard" >> "$STATUS"
  exit 1
fi
