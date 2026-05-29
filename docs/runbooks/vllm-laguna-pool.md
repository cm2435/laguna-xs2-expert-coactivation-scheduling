# vLLM Laguna + Pool Runbook

## Start vLLM

```bash
scripts/run_vllm_laguna_server.sh
```

## Smoke vLLM Directly

```bash
curl -s http://127.0.0.1:8791/v1/models
```

## Smoke Through Pool

```bash
POOLSIDE_API_KEY=dummy pool exec \
  --sandbox disabled \
  --api-url http://127.0.0.1:8791/v1 \
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
