#!/usr/bin/env bash
set -euo pipefail

REGISTRY="tasks/registry_balanced_100_train80.jsonl"
LIMIT="80"
MAX_TURNS="15"
TEACHER_API_URL="http://127.0.0.1:8791/v1"
TEACHER_MODEL="laguna"
STUDENT_MODEL=""
SEQ_LEN="8192"
SFT_MAX_STEPS="500"
SFT_BATCH_SIZE="1"
SFT_LR="5e-5"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SKIP_TRAIN="0"
DISABLE_THINKING="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --teacher-api-url) TEACHER_API_URL="$2"; shift 2 ;;
    --teacher-model) TEACHER_MODEL="$2"; shift 2 ;;
    --student-model) STUDENT_MODEL="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --sft-max-steps) SFT_MAX_STEPS="$2"; shift 2 ;;
    --sft-batch-size) SFT_BATCH_SIZE="$2"; shift 2 ;;
    --sft-lr) SFT_LR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN="1"; shift ;;
    --disable-thinking) DISABLE_THINKING="1"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NIGHTLY_DIR="runs/nightly/rollout_to_sft_${RUN_ID}"
LOG_DIR="${NIGHTLY_DIR}/logs"
ROLLOUT_DIR="runs/coding_harness_${RUN_ID}"
SANDBOX_DIR="sandboxes/coding_harness_${RUN_ID}"
SUMMARY_PATH="${NIGHTLY_DIR}/rollout_summary.json"
SFT_JSONL="data/sft/rollout_sft_${RUN_ID}.jsonl"
SFT_RUN_DIR="runs/sft/${RUN_ID}"

if [[ -e "$NIGHTLY_DIR" ]]; then
  echo "run directory already exists: $NIGHTLY_DIR" >&2
  exit 1
fi
mkdir -p "$LOG_DIR" "$(dirname "$SFT_JSONL")"

{
  echo "RUN_ID=${RUN_ID}"
  echo "REGISTRY=${REGISTRY}"
  echo "LIMIT=${LIMIT}"
  echo "MAX_TURNS=${MAX_TURNS}"
  echo "TEACHER_API_URL=${TEACHER_API_URL}"
  echo "STUDENT_MODEL=${STUDENT_MODEL}"
  echo "SEQ_LEN=${SEQ_LEN}"
  echo "DISABLE_THINKING=${DISABLE_THINKING}"
} > "${NIGHTLY_DIR}/env.txt"
printf '%q ' "$0" "$@" > "${NIGHTLY_DIR}/command.sh"
echo >> "${NIGHTLY_DIR}/command.sh"

status_failed() {
  local stage="$1"
  local message="$2"
  python - "$NIGHTLY_DIR/status.json" "$stage" "$message" <<'PY'
import json, sys
path, stage, message = sys.argv[1:4]
open(path, "w").write(json.dumps({"success": False, "failed_stage": stage, "message": message}, indent=2) + "\n")
PY
}

trap 'status_failed "unknown" "unexpected failure at line ${LINENO}"' ERR

echo "Checking teacher endpoint..."
curl -fsS "${TEACHER_API_URL%/}/models" > "${LOG_DIR}/00_teacher_models.json"

echo "Preparing repo templates..."
uv run --no-sync python scripts/prepare_repo_templates.py \
  --registry "$REGISTRY" \
  --limit "$LIMIT" \
  > "${LOG_DIR}/01_prepare_templates.log" 2>&1

echo "Running teacher rollouts..."
uv run --no-sync python scripts/run_coding_swebench_batch.py \
  --registry "$REGISTRY" \
  --api-url "$TEACHER_API_URL" \
  --model "$TEACHER_MODEL" \
  --output-dir "$ROLLOUT_DIR" \
  --sandbox-root "$SANDBOX_DIR" \
  --limit "$LIMIT" \
  --max-turns "$MAX_TURNS" \
  --temperature 0.0 \
  > "${LOG_DIR}/02_rollouts.log" 2>&1

echo "Summarizing rollouts..."
uv run --no-sync python scripts/summarize_coding_rollouts.py \
  --runs-dir "$ROLLOUT_DIR" \
  --sandboxes-dir "$SANDBOX_DIR" \
  --output "$SUMMARY_PATH" \
  > "${LOG_DIR}/03_summarize.log" 2>&1

echo "Building SFT JSONL..."
uv run --no-sync python scripts/build_sft_from_rollouts.py \
  --runs-dir "$ROLLOUT_DIR" \
  --sandboxes-dir "$SANDBOX_DIR" \
  --output "$SFT_JSONL" \
  --max-turns "$MAX_TURNS" \
  > "${LOG_DIR}/04_build_sft.log" 2>&1

VALIDATE_ARGS=()
if [[ "$DISABLE_THINKING" == "1" ]]; then
  VALIDATE_ARGS+=(--disable-thinking)
fi
echo "Validating SFT JSONL..."
uv run --no-sync python scripts/validate_sft_dataset.py \
  --input "$SFT_JSONL" \
  --output "${NIGHTLY_DIR}/sft_validation.json" \
  --seq-len "$SEQ_LEN" \
  "${VALIDATE_ARGS[@]}" \
  > "${LOG_DIR}/05_validate_sft.log" 2>&1

if [[ "$SKIP_TRAIN" == "0" ]]; then
  if [[ -z "$STUDENT_MODEL" ]]; then
    echo "--student-model is required unless --skip-train is passed" >&2
    exit 2
  fi
  echo "Training dense SFT..."
  TRAIN_ARGS=()
  if [[ "$DISABLE_THINKING" == "1" ]]; then
    TRAIN_ARGS+=(--disable-thinking)
  fi
  uv run --no-sync python scripts/train_dense_sft.py \
    --model "$STUDENT_MODEL" \
    --dataset "$SFT_JSONL" \
    --output-dir "$SFT_RUN_DIR" \
    --seq-len "$SEQ_LEN" \
    --max-steps "$SFT_MAX_STEPS" \
    --batch-size "$SFT_BATCH_SIZE" \
    --lr "$SFT_LR" \
    "${TRAIN_ARGS[@]}" \
    > "${LOG_DIR}/06_train_sft.log" 2>&1
fi

python - "$NIGHTLY_DIR/artifacts.json" "$RUN_ID" "$ROLLOUT_DIR" "$SANDBOX_DIR" "$SFT_JSONL" "$SFT_RUN_DIR" <<'PY'
import json, sys
path, run_id, rollout_dir, sandbox_dir, sft_jsonl, sft_run_dir = sys.argv[1:7]
open(path, "w").write(json.dumps({
    "run_id": run_id,
    "rollout_dir": rollout_dir,
    "sandbox_dir": sandbox_dir,
    "sft_jsonl": sft_jsonl,
    "sft_run_dir": sft_run_dir,
}, indent=2) + "\n")
PY
python - "$NIGHTLY_DIR/status.json" <<'PY'
import json, sys
open(sys.argv[1], "w").write(json.dumps({"success": True}, indent=2) + "\n")
PY

echo "Done: ${NIGHTLY_DIR}"
