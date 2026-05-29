# vLLM Laguna + Pool Runbook

## Start vLLM

```bash
scripts/run_vllm_laguna_server.sh
```

The checked-in config is the B300-safe rollout path we verified:

```text
attention-backend: FLASH_ATTN
moe-backend: triton
VLLM_USE_FLASHINFER_SAMPLER=0
enable_thinking: false
served-model-name: laguna
```

These settings avoid the FlashInfer B300 JIT paths that failed on the Prime
image's mixed CUDA 12.8 / CUDA 13 toolchain.

## Smoke vLLM Directly

```bash
curl -s http://127.0.0.1:8791/v1/models
```

```bash
curl -s http://127.0.0.1:8791/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"laguna","messages":[{"role":"user","content":"Reply with the single word READY."}],"max_tokens":16,"temperature":0}'
```

## Smoke Through Pool

```bash
POOLSIDE_API_KEY=dummy pool exec \
  --sandbox disabled \
  --api-url http://127.0.0.1:8792/v1 \
  -a default \
  -p "Reply with the single word READY." \
  -o json
```

## Smoke Through Recording Proxy

```bash
uv run python scripts/run_recording_openai_proxy.py \
  --listen-port 8792 \
  --upstream-base-url http://127.0.0.1:8791/v1 \
  --output-dir runs/proxy_vllm_smoke
```

Then:

```bash
POOLSIDE_API_KEY=dummy pool exec \
  --sandbox disabled \
  --api-url http://127.0.0.1:8792/v1 \
  -a default \
  -p "Reply with the single word READY." \
  -o json
```
