# RFC 001: Mobile Targets And Memory Budget

**Status:** Active planning.

## Purpose

Define what "mobile serving" means for this project, what hardware we are targeting, and what memory budget the dense model has to meet.

The key constraint is simple: the BF16 checkpoint is not mobile-sized. A successful demo must be framed around an aggressively constrained inference profile rather than full-quality desktop serving.

## Model Size And KV Budget

Current dense checkpoint:

```text
BF16 weights: approximately 5.6 GB
parameter count implied by BF16 weights: approximately 2.8B parameters
```

Approximate KV cache size:

```text
per layer per token:
    2 tensors (K,V) * 8 KV heads * 128 head_dim * 2 bytes BF16
    = 4096 bytes/layer/token

context 512, all 40 layers:
    approximately 80-84 MB

context 1024, 10 full layers + 30 sliding-window layers capped at 512:
    approximately 105 MB

context 2048, 10 full layers + 30 sliding-window layers capped at 512:
    approximately 145-150 MB

context 4096, 10 full layers + 30 sliding-window layers capped at 512:
    approximately 230 MB
```

The KV cache is not the main blocker at short context. The blocker is weight memory, load-time duplication, runtime allocator overhead, and app sandbox pressure.

## Device Tiers

### Tier 0: Native Mac Smoke

Goal: verify that the runtime can load and run the architecture.

```text
hardware: MacBook Pro
memory target: loose
context: 128-512
batch: 1
purpose: catch export/runtime issues before pretending this is mobile
```

### Tier 1: Mac Simulated Phone

Goal: make the Mac run look like a phone deployment.

```text
hardware: MacBook Pro
memory target: peak RSS < 3.5 GB, stretch < 3.0 GB
context: 512
batch: 1
prefill chunk: <= 128 if supported
decode: greedy first
purpose: prove the serving artifact is plausibly phone-shaped
```

This is the first real milestone.

### Tier 2: 6 GB iPhone Stress Target

Examples:

```text
iPhone 12 Pro Max: 6 GB RAM
iPhone 14 Pro Max: 6 GB RAM
```

Conclusion:

```text
BF16: no
8-bit: likely no, once runtime overhead and app memory pressure are included
4-bit: maybe, with short context and careful loading
3-bit / 2-bit: more plausible, but likely lower quality and more runtime work
```

These devices are useful as stress tests, not the default success target.

### Tier 3: Practical Real Phone Target

Examples:

```text
iPhone 15 Pro / 15 Pro Max
iPhone 16 generation
high-end Android devices with 8-12 GB RAM
```

Conclusion:

```text
4-bit serving is much more plausible here.
Short context is still mandatory.
Thermal and sustained decode speed remain open questions.
```

## Quantization Targets

Approximate weight footprint:

```text
BF16: 5.6 GB
INT8 / FP8 class: ~2.8 GB before metadata/runtime overhead
Q4 class: ~1.4 GB raw, likely 1.6-2.2 GB with scales/packing/runtime overhead
Q3 class: ~1.1-1.6 GB practical range
Q2 class: lower memory, larger quality risk
```

The first mobile artifact should target Q4. Q3 is the fallback if Q4 cannot fit under the simulated-phone budget.

## Metrics

Primary:

```text
peak RSS during load
peak RSS during 32-token decode
time to first token (TTFT)
decode tokens/sec
model load time
```

Secondary:

```text
binary/artifact size
context length supported under budget
allocator fragmentation / weight duplication signals
energy/thermal behavior on real device, when available
```

## Pass/Fail Gates

Pass:

```text
loads under 3.5 GB peak RSS in simulated-phone mode
generates 32 tokens without memory spikes
does not duplicate weights after warmup
reports TTFT and decode tokens/sec
has a documented next step toward iOS or Android packaging
```

Fail:

```text
BF16-only serving path
requires > 5 GB peak RSS for batch=1, context=512
requires server-side Python to be part of the "mobile" demo
cannot tokenize and generate from the real checkpoint artifact
```

## Decision

The near-term target is not "run well on iPhone 12 Pro Max." The near-term target is "produce a quantized, phone-shaped serving artifact on Mac with measured memory and latency, using a runtime that can plausibly move to iOS/Android."
