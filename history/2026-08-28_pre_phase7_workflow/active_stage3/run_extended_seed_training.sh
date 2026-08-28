#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_extended_seed_training.sh --seed SEED --model-kind direct|masked_conditional [--preflight] [--resume PATH]

Environment:
  PYTHON_BIN  Python executable to use. Defaults to ~/miniconda3/envs/gdm-stage39/bin/python.
USAGE
}

SEED=""
MODEL_KIND=""
PREFLIGHT=0
RESUME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --model-kind)
      MODEL_KIND="${2:-}"
      shift 2
      ;;
    --preflight)
      PREFLIGHT=1
      shift
      ;;
    --resume)
      RESUME="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$SEED" in
  2026070114|2026070115|2026070116|2026070117|2026070118|2026070119|2026070120) ;;
  *)
    echo "Unsupported Stage 3.9 seed: ${SEED:-<empty>}" >&2
    exit 2
    ;;
esac

case "$MODEL_KIND" in
  direct|masked_conditional) ;;
  *)
    echo "Unsupported model kind: ${MODEL_KIND:-<empty>}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPLEMENTATION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/gdm-stage39/bin/python}"
TRAINING_SCRIPT="$IMPLEMENTATION_ROOT/scripts/64_train_phase6e_e_stage3.py"
TRAINING_CONFIG="$IMPLEMENTATION_ROOT/configs/training_phase6e_e_stage39_seed${SEED}.yaml"
STAGE38_FREEZE="$IMPLEMENTATION_ROOT/artifacts/phase6e-e-stage38-training/training_freeze.json"
DEVELOPMENT_FREEZE="$IMPLEMENTATION_ROOT/artifacts/datasets/phase6e-e-stage3-development/dataset_freeze.json"

for required_path in \
  "$PYTHON_BIN" \
  "$TRAINING_SCRIPT" \
  "$TRAINING_CONFIG" \
  "$STAGE38_FREEZE" \
  "$DEVELOPMENT_FREEZE"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required extended-seed training path is missing: $required_path" >&2
    exit 1
  fi
done

args=(
  "$TRAINING_SCRIPT"
  "--config" "$TRAINING_CONFIG"
  "--model-kind" "$MODEL_KIND"
)
if [[ "$PREFLIGHT" -eq 1 ]]; then
  args+=("--preflight")
fi
if [[ -n "$RESUME" ]]; then
  args+=("--resume" "$RESUME")
fi

cd "$IMPLEMENTATION_ROOT"
export GDM_STAGE3_ACTIVE_ENTRY=1
exec "$PYTHON_BIN" "${args[@]}"
