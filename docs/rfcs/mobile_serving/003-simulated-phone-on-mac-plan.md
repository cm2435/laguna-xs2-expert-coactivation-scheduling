# RFC 003: Simulated Phone On Mac Plan

**Status:** Active planning.

## Purpose

Define the first measurable mobile-serving target: a phone-shaped run on Mac.

This exists because iOS Simulator proves app integration, not real phone memory behavior. A MacBook Pro can still be useful if we impose the constraints we expect a phone deployment to face.

## What This Is

Simulated phone means:

```text
same checkpoint artifact intended for mobile
batch size = 1
short context
strict peak memory budget
no server process pretending to be on-device
no Python dependency in the final measured path
```

It is a serving profile, not a claim that the Mac is a phone.

## What This Is Not

Not acceptable as the headline:

```text
running a Python Transformers model on Mac
running an unconstrained desktop binary
using a 16 GB+ memory footprint and calling it mobile
measuring only artifact size without measuring peak RSS
```

## Profiles

### Profile A: Native Mac Smoke

Use this first to avoid debugging memory and export at the same time.

```text
context: 128
batch: 1
generation: 8-32 tokens
dtype/quant: whatever exports first
memory target: none
```

Pass condition:

```text
model artifact loads and produces non-empty text
```

### Profile B: 6 GB Phone Stress

This approximates iPhone 12 Pro Max / iPhone 14 Pro Max pressure.

```text
context: 512
batch: 1
generation: 32 tokens
quant: Q4 target, Q3 fallback
peak RSS target: < 3.5 GB
stretch target: < 3.0 GB
```

Pass condition:

```text
32-token decode without exceeding memory target
TTFT and tokens/sec recorded
```

### Profile C: 8-12 GB Practical Phone

This approximates newer high-memory devices.

```text
context: 512-1024
batch: 1
generation: 64 tokens
quant: Q4 target
peak RSS target: < 5.0 GB
```

Pass condition:

```text
longer decode remains stable and has no large memory growth
```

## Measurement Plan

Collect:

```text
artifact size
model load time
peak RSS during load
RSS after warmup
peak RSS during decode
TTFT
decode tokens/sec
generated token count
context length
quantization format
runtime backend
```

Mac measurement tools:

```bash
/usr/bin/time -l <runner command>
vmmap <pid>
top -pid <pid>
memory_pressure
```

For app shells, record:

```text
Xcode memory gauge
console timings
binary and asset sizes
```

## Test Prompt Set

Use tiny prompts that exercise the tokenizer and causal path without turning quality into the bottleneck.

```text
1. "Write a Python function that returns the sum of a list."
2. "Complete this code:\n\ndef add(a, b):\n"
3. "Explain in one sentence what a unit test is."
4. "Return only valid JSON with a field named status."
```

Quality is not the primary metric here. The model is known to need more training. The mobile-serving milestone only requires stable generation.

## Output Artifact

Each run should write:

```text
runs/mobile_serving/<runtime>/<timestamp>/
    config.json
    prompt.txt
    generated.txt
    metrics.json
    memory.txt
```

Required `metrics.json` fields:

```json
{
  "runtime": "executorch",
  "artifact": "path-or-hf-id",
  "quantization": "q4",
  "context_tokens": 512,
  "generated_tokens": 32,
  "peak_rss_mb": 0,
  "rss_after_warmup_mb": 0,
  "model_load_ms": 0,
  "ttft_ms": 0,
  "decode_tokens_per_second": 0
}
```

## Gotchas

### iOS Simulator Is Not Enough

iOS Simulator uses Mac resources and does not replicate iPhone memory pressure. It is still valuable for packaging, API shape, and UI integration.

### Hidden Weight Copies

The raw Q4 model might be small enough, but the runtime can briefly hold packed and unpacked copies. Peak RSS, not artifact size, decides whether this is plausible.

### Context Creep

The Laguna architecture supports huge context, but mobile does not. Keep the mobile proof at 512 tokens until everything else works.

### Recompute Decode Can Mislead

If the first runner recomputes the full context every token, tokens/sec will be poor. That is acceptable for portability smoke, but not for the final serving claim.

## Decision

The first mobile proof is Profile B: batch=1, context=512, 32-token greedy decode, peak RSS below 3.5 GB, and measured TTFT/tokens/sec.
