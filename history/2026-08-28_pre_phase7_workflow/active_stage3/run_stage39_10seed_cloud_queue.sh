#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPLEMENTATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/gdm-stage39/bin/python}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs configured. Set GPU_LIST, e.g. GPU_LIST=0,1,2,3." >&2
  exit 2
fi

SEEDS=(2026070114 2026070115 2026070116 2026070117 2026070118 2026070119 2026070120)
KINDS=(masked_conditional masked_conditional masked_conditional masked_conditional masked_conditional masked_conditional masked_conditional direct direct direct direct direct direct direct)
JOB_SEEDS=(2026070114 2026070115 2026070116 2026070117 2026070118 2026070119 2026070120 2026070114 2026070115 2026070116 2026070117 2026070118 2026070119 2026070120)

ARTIFACT_ROOT="$IMPLEMENTATION_ROOT/artifacts/phase6e-e-stage39-10seed-training"
LOG_DIR="$ARTIFACT_ROOT/logs"
STATUS_FILE="$ARTIFACT_ROOT/cloud_queue_status.tsv"
PID_FILE="$ARTIFACT_ROOT/cloud_queue_pids.tsv"
mkdir -p "$LOG_DIR"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

echo -e "timestamp\tbatch\tgpu\tmodel_kind\tseed\tevent\texit_code\tlog" > "$STATUS_FILE"
: > "$PID_FILE"

launch_job() {
  local batch="$1"
  local gpu="$2"
  local model_kind="$3"
  local seed="$4"
  local log_path="$LOG_DIR/${model_kind}-seed${seed}.log"
  local run_dir="$ARTIFACT_ROOT/${model_kind}-seed${seed}"
  local resume_args=()

  if [[ -f "$run_dir/latest.pt" ]]; then
    resume_args=("--resume" "$run_dir/latest.pt")
  elif [[ -f "$run_dir/metrics.jsonl" ]]; then
    echo "Existing run has metrics but no latest checkpoint: $run_dir" >&2
    return 2
  fi

  echo -e "$(timestamp)\t$batch\t$gpu\t$model_kind\t$seed\tstarted\t\t$log_path" | tee -a "$STATUS_FILE"
  (
    cd "$IMPLEMENTATION_ROOT"
    echo "[stage39] start $(timestamp) gpu=$gpu kind=$model_kind seed=$seed"
    echo "[stage39] python=$PYTHON_BIN"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHON_BIN="$PYTHON_BIN" \
      "$SCRIPT_DIR/run_extended_seed_training.sh" \
      --seed "$seed" \
      --model-kind "$model_kind" \
      "${resume_args[@]}"
    code=$?
    echo "[stage39] finish $(timestamp) gpu=$gpu kind=$model_kind seed=$seed exit=$code"
    exit "$code"
  ) > "$log_path" 2>&1 &
  local pid=$!
  LAUNCHED_PID="$pid"
  echo -e "$pid\t$gpu\t$model_kind\t$seed\t$log_path" >> "$PID_FILE"
}

total_jobs="${#JOB_SEEDS[@]}"
batch=0
failures=0

for ((start=0; start<total_jobs; start+=${#GPUS[@]})); do
  batch=$((batch + 1))
  pids=()
  labels=()
  for ((offset=0; offset<${#GPUS[@]}; offset++)); do
    index=$((start + offset))
    if [[ "$index" -ge "$total_jobs" ]]; then
      break
    fi
    gpu="${GPUS[$offset]}"
    model_kind="${KINDS[$index]}"
    seed="${JOB_SEEDS[$index]}"
    if launch_job "$batch" "$gpu" "$model_kind" "$seed"; then
      pids+=("$LAUNCHED_PID")
      labels+=("$gpu:$model_kind:$seed")
    else
      failures=$((failures + 1))
      echo -e "$(timestamp)\t$batch\t$gpu\t$model_kind\t$seed\tlaunch_failed\t2\t" | tee -a "$STATUS_FILE"
    fi
  done

  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    label="${labels[$i]}"
    IFS=':' read -r gpu model_kind seed <<< "$label"
    if wait "$pid"; then
      code=0
    else
      code=$?
      failures=$((failures + 1))
    fi
    log_path="$LOG_DIR/${model_kind}-seed${seed}.log"
    echo -e "$(timestamp)\t$batch\t$gpu\t$model_kind\t$seed\tfinished\t$code\t$log_path" | tee -a "$STATUS_FILE"
  done
done

echo "[stage39] queue finished at $(timestamp), failures=$failures" | tee -a "$LOG_DIR/cloud_queue.log"
exit "$failures"
