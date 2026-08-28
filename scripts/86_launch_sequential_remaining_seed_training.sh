#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/GDM_Paper/implementation}"
PYTHON_BIN="${PYTHON_BIN:-/home/linchen/miniconda3/envs/gdm-stage39/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/phase6f-sequential-conditional-training}"
LOG_DIR="$ROOT/$OUTPUT_ROOT/queue_logs"
mkdir -p "$LOG_DIR"

run_seed() {
  local gpu="$1"
  local seed="$2"
  local config="$3"
  local run_dir="$ROOT/$OUTPUT_ROOT/sequential_conditional-seed${seed}"
  local seed_log="$LOG_DIR/seed${seed}_gpu${gpu}.log"

  echo "[$(date -Is)] seed=${seed} gpu=${gpu} config=${config}" | tee -a "$seed_log"
  if [[ -f "$run_dir/latest.pt" ]] && [[ -f "$run_dir/metrics.jsonl" ]] && grep -q '"step": 20000' "$run_dir/metrics.jsonl"; then
    echo "[$(date -Is)] seed=${seed} already complete; skipping." | tee -a "$seed_log"
    return 0
  fi

  local resume_args=()
  if [[ -f "$run_dir/latest.pt" ]]; then
    resume_args=(--resume "$run_dir/latest.pt")
    echo "[$(date -Is)] seed=${seed} resuming from $run_dir/latest.pt" | tee -a "$seed_log"
  fi

  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON_BIN" \
      scripts/84_train_sequential_conditional_gnn.py \
      --config "$config" \
      --max-steps 20000 \
      --selection-instances 64 \
      --output-root "$OUTPUT_ROOT" \
      "${resume_args[@]}"
  ) >> "$seed_log" 2>&1
  local status=$?
  echo "[$(date -Is)] seed=${seed} finished status=${status}" | tee -a "$seed_log"
  return "$status"
}

worker() {
  local gpu="$1"
  shift
  local item seed config
  for item in "$@"; do
    seed="${item%%:*}"
    config="${item#*:}"
    if ! run_seed "$gpu" "$seed" "$config"; then
      echo "[$(date -Is)] seed=${seed} failed on gpu=${gpu}; continuing queue." | tee -a "$LOG_DIR/worker_gpu${gpu}.log"
    fi
  done
}

status() {
  cd "$ROOT"
  echo "=== sequential training processes ==="
  ps -u "${USER:-linchen}" -o pid,etime,stat,args | grep 84_train_sequential_conditional_gnn | grep -v grep || true
  echo "=== worker logs ==="
  ls -lh "$LOG_DIR"/worker_gpu*.log 2>/dev/null || true
  echo "=== completed latest checkpoints ==="
  find "$OUTPUT_ROOT" -maxdepth 2 -type f -name latest.pt | sort || true
}

start() {
  cd "$ROOT"
  nohup bash "$0" worker 0 \
    2026070111:configs/training_phase6e_e_stage3_pilot.yaml \
    2026070115:configs/training_phase6e_e_stage39_seed2026070115.yaml \
    2026070119:configs/training_phase6e_e_stage39_seed2026070119.yaml \
    > "$LOG_DIR/worker_gpu0.log" 2>&1 &
  echo "worker_gpu0_pid=$!"

  nohup bash "$0" worker 1 \
    2026070112:configs/training_phase6e_e_stage38_seed2026070112.yaml \
    2026070116:configs/training_phase6e_e_stage39_seed2026070116.yaml \
    > "$LOG_DIR/worker_gpu1.log" 2>&1 &
  echo "worker_gpu1_pid=$!"

  nohup bash "$0" worker 2 \
    2026070113:configs/training_phase6e_e_stage38_seed2026070113.yaml \
    2026070117:configs/training_phase6e_e_stage39_seed2026070117.yaml \
    > "$LOG_DIR/worker_gpu2.log" 2>&1 &
  echo "worker_gpu2_pid=$!"

  nohup bash "$0" worker 3 \
    2026070118:configs/training_phase6e_e_stage39_seed2026070118.yaml \
    2026070120:configs/training_phase6e_e_stage39_seed2026070120.yaml \
    > "$LOG_DIR/worker_gpu3.log" 2>&1 &
  echo "worker_gpu3_pid=$!"
}

case "${1:-start}" in
  start)
    start
    ;;
  worker)
    shift
    worker "$@"
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 [start|status|worker ...]" >&2
    exit 2
    ;;
esac
