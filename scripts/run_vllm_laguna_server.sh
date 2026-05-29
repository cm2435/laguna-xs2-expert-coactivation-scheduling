#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

uv run --no-sync vllm serve --config configs/vllm_laguna_xs2.yaml
