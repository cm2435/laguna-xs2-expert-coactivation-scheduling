#!/usr/bin/env bash
set -euo pipefail

REGISTRY="tasks/registry_balanced_100_train80.jsonl"
VALIDATION_REGISTRY="tasks/registry_balanced_100_val20.jsonl"
LIMIT="80"
MAX_TURNS="15"
TEACHER_API_URL="http://127.0.0.1:8791/v1"
TEACHER_MODEL="laguna"
STUDENT_MODEL=""
SEQ_LEN="8192"
SFT_MAX_STEPS="1000"
SFT_BATCH_SIZE="1"
SFT_LR="5e-5"
SFT_WEIGHT="1.0"
KD_WEIGHT="0.3"
KD_TEMPERATURE="1.0"
TOP_LOGPROBS="20"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SKIP_TRAIN="0"
SKIP_ROLLOUTS="0"
DRY_RUN_LOGPROBS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="$2"; shift 2 ;;
    --validation-registry|--val-registry) VALIDATION_REGISTRY="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --teacher-api-url) TEACHER_API_URL="$2"; shift 2 ;;
    --teacher-model) TEACHER_MODEL="$2"; shift 2 ;;
    --student-model) STUDENT_MODEL="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --sft-max-steps) SFT_MAX_STEPS="$2"; shift 2 ;;
    --sft-batch-size) SFT_BATCH_SIZE="$2"; shift 2 ;;
    --sft-lr) SFT_LR="$2"; shift 2 ;;
    --sft-weight) SFT_WEIGHT="$2"; shift 2 ;;
    --kd-weight) KD_WEIGHT="$2"; shift 2 ;;
    --kd-temperature) KD_TEMPERATURE="$2"; shift 2 ;;
    --top-logprobs) TOP_LOGPROBS="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN="1"; shift ;;
    --skip-rollouts) SKIP_ROLLOUTS="1"; shift ;;
    --dry-run-logprobs) DRY_RUN_LOGPROBS="1"; shift ;;
    --capture-logprobs) shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NIGHTLY_DIR="runs/nightly/rollout_to_sft_kd_${RUN_ID}"
LOG_DIR="${NIGHTLY_DIR}/logs"
ROLLOUT_DIR="runs/coding_harness_${RUN_ID}"
SANDBOX_DIR="sandboxes/coding_harness_${RUN_ID}"
SUMMARY_PATH="${NIGHTLY_DIR}/rollout_summary.json"
SFT_JSONL="data/sft/rollout_sft_${RUN_ID}.jsonl"
KD_JSONL="data/kd/teacher_logprobs_${RUN_ID}.jsonl"
SFT_RUN_DIR="runs/sft_kd/${RUN_ID}"

if [[ -e "$NIGHTLY_DIR" ]]; then
  echo "run directory already exists: $NIGHTLY_DIR" >&2
  exit 1
fi
mkdir -p "$LOG_DIR" "$(dirname "$SFT_JSONL")" "$(dirname "$KD_JSONL")"

{
  echo "RUN_ID=${RUN_ID}"
  echo "REGISTRY=${REGISTRY}"
  echo "VALIDATION_REGISTRY=${VALIDATION_REGISTRY}"
  echo "LIMIT=${LIMIT}"
  echo "MAX_TURNS=${MAX_TURNS}"
  echo "TEACHER_API_URL=${TEACHER_API_URL}"
  echo "STUDENT_MODEL=${STUDENT_MODEL}"
  echo "SEQ_LEN=${SEQ_LEN}"
  echo "TOP_LOGPROBS=${TOP_LOGPROBS}"
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

if [[ "$SKIP_ROLLOUTS" == "0" ]]; then
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
fi

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

echo "Capturing teacher logprobs..."
LOGPROB_ARGS=()
if [[ "$DRY_RUN_LOGPROBS" == "1" ]]; then
  LOGPROB_ARGS+=(--dry-run)
fi
uv run --no-sync python scripts/capture_teacher_logprobs.py \
  --input "$SFT_JSONL" \
  --output "$KD_JSONL" \
  --teacher-api-url "$TEACHER_API_URL" \
  --model "$TEACHER_MODEL" \
  --top-logprobs "$TOP_LOGPROBS" \
  "${LOGPROB_ARGS[@]}" \
  > "${LOG_DIR}/05_capture_logprobs.log" 2>&1

if [[ "$SKIP_TRAIN" == "0" ]]; then
  if [[ -z "$STUDENT_MODEL" ]]; then
    echo "--student-model is required unless --skip-train is passed" >&2
    exit 2
  fi
  echo "Training dense SFT+KD..."
  uv run --no-sync python scripts/train_dense_sft.py \
    --model "$STUDENT_MODEL" \
    --dataset "$SFT_JSONL" \
    --kd-dataset "$KD_JSONL" \
    --output-dir "$SFT_RUN_DIR" \
    --seq-len "$SEQ_LEN" \
    --max-steps "$SFT_MAX_STEPS" \
    --batch-size "$SFT_BATCH_SIZE" \
    --lr "$SFT_LR" \
    --sft-weight "$SFT_WEIGHT" \
    --kd-weight "$KD_WEIGHT" \
    --kd-temperature "$KD_TEMPERATURE" \
    > "${LOG_DIR}/06_train_sft_kd.log" 2>&1
fi

python - "$NIGHTLY_DIR/artifacts.json" "$RUN_ID" "$ROLLOUT_DIR" "$SANDBOX_DIR" "$SFT_JSONL" "$KD_JSONL" "$SFT_RUN_DIR" <<'PY'
import json, sys
path, run_id, rollout_dir, sandbox_dir, sft_jsonl, kd_jsonl, sft_run_dir = sys.argv[1:8]
open(path, "w").write(json.dumps({
    "run_id": run_id,
    "rollout_dir": rollout_dir,
    "sandbox_dir": sandbox_dir,
    "sft_jsonl": sft_jsonl,
    "kd_jsonl": kd_jsonl,
    "sft_kd_run_dir": sft_run_dir,
}, indent=2) + "\n")
PY
python - "$NIGHTLY_DIR/status.json" <<'PY'
import json, sys
open(sys.argv[1], "w").write(json.dumps({"success": True}, indent=2) + "\n")
PY

echo "Done: ${NIGHTLY_DIR}"
