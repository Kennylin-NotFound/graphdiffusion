#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/GDM_Paper/implementation}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/gdm-stage39/bin/python}"
OUT_BASE="${OUT_BASE:-artifacts/phase6f-decoding-policy-full}"
OUT="$ROOT/$OUT_BASE"
LOG_DIR="$OUT/logs"
STATUS="$OUT/queue_status.tsv"
SEEDS="${SEEDS:-2026070111,2026070112,2026070113,2026070114,2026070115,2026070116,2026070117,2026070118,2026070119,2026070120}"
METHODS="${METHODS:-greedy,direct_b64_t1,direct_b64_t1_ens,sequential_b64_t1,sequential_b64_t1_mix,masked_k8_t1,masked_k8_t1_mix,masked_k8_t1_ens,masked_k16_t1}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

"$PYTHON" -m py_compile scripts/92_probe_neural_decoding_enhancements.py

{
  echo -e "event\tutc\tsetting\tgpu\tpid\tstatus\toutput_root"
  echo -e "launch\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\tall\t-\t$$\trunning\t$OUT_BASE"
} > "$STATUS"

run_setting() {
  local setting="$1"
  local gpu="$2"
  local setting_out="$OUT_BASE/$setting"
  local log="$LOG_DIR/${setting}_gpu${gpu}.log"

  if pgrep -af "92_probe_neural_decoding_enhancements.py all --settings ${setting}" >/dev/null; then
    echo -e "skip\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t${setting}\t${gpu}\t-\talready_running\t${setting_out}" >> "$STATUS"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" \
    scripts/92_probe_neural_decoding_enhancements.py all \
    --settings "$setting" \
    --seeds "$SEEDS" \
    --methods "$METHODS" \
    --device cuda \
    --output-root "$setting_out" \
    --fallback-max-search-nodes 30000 \
    --no-skip-missing-datasets \
    > "$log" 2>&1 &

  local pid="$!"
  echo -e "start\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t${setting}\t${gpu}\t${pid}\trunning\t${setting_out}" >> "$STATUS"
}

run_setting "sealed_id" 0
run_setting "controlled_shift" 1
run_setting "realistic_profile" 2
run_setting "cross_scale" 3

echo -e "finish_launch\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\tall\t-\t$$\tok\t$OUT_BASE" >> "$STATUS"
