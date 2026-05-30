# RFC 002: ExecuTorch Serving Plan

**Status:** Active planning.

## Purpose

Define the primary implementation path for serving the custom dense Laguna checkpoint on mobile-class runtimes.

ExecuTorch is the recommended primary path because our model is not a stock Llama checkpoint. We need a runtime that starts from PyTorch modules, tolerates custom model code, and can eventually target iOS/Android.

## Why ExecuTorch

Pros:

```text
PyTorch-first export path
supports custom module graphs better than LLM-specific runtimes
iOS and Android deployment story
CPU backend available for first smoke
can start with fixed-shape forward before optimizing generation
```

Cons:

```text
generation loop and KV cache may need custom work
remote-code HF model may not export without surgery
low-bit LLM quantization path may be less turnkey than llama.cpp
custom attention variants may require lowering fixes
```

## Target Architecture

Pipeline:

```text
HF checkpoint
  -> PyTorch model load
  -> exportable forward wrapper
  -> torch.export
  -> ExecuTorch .pte artifact
  -> Mac runner
  -> iOS Simulator / iOS app shell
  -> real device once memory is credible
```

Initial serving mode:

```text
batch size: 1
context: 128 first, then 512
decode: greedy
sampling: disabled
KV cache: optional in first export; required for meaningful latency
quantization: start unquantized for export smoke, then Q8/Q4
```

## Components We Need To Write

### 1. Export Wrapper

Domain: Python / PyTorch.

Responsibilities:

```text
load cm2435-new/laguna-xs2-dense-k8-recon-2k
wrap the HF remote-code model with a fixed-shape forward
avoid generation APIs during export
return logits for a fixed input_ids tensor
disable unsupported dynamic cache behavior for first smoke
```

The first wrapper can be intentionally dumb. It only needs:

```text
input_ids: [1, seq_len]
attention_mask: optional if the model requires it
logits: [1, seq_len, vocab]
```

### 2. Export Script

Domain: Python / ExecuTorch.

Responsibilities:

```text
call torch.export on the wrapper
lower to ExecuTorch
write a .pte artifact
save tokenizer/config metadata beside it
record export shape and dtype
```

### 3. Mac Runner

Domain: C++ or Python runner, depending on fastest available ExecuTorch sample.

Responsibilities:

```text
load .pte
feed token ids
return logits
implement a 1-step next-token loop
print generated token ids and timing
```

### 4. Greedy Decode Loop

Domain: host runtime code.

Responsibilities:

```text
tokenize prompt
prefill context
decode N tokens greedily
detokenize result
measure TTFT and tokens/sec
```

First pass may recompute the whole context each token. That proves portability but not speed. Second pass should add KV.

### 5. Quantized Artifact Path

Domain: PyTorch export / ExecuTorch quantization.

Responsibilities:

```text
produce Q8 baseline artifact
produce Q4 target artifact if supported
verify artifact size and peak RSS
compare logits drift on a tiny prompt set
```

## Implementation Spikes

### Spike A: Load And Forward

Goal:

```text
load the HF checkpoint on Mac or GPU VM
run one forward pass with seq_len=128
verify logits shape and sane non-NaN output
```

Exit criteria:

```text
script can run from clean checkout
prints model dtype, peak memory, logits shape
```

### Spike B: Export Fixed-Shape Forward

Goal:

```text
export seq_len=128 forward to ExecuTorch
```

Exit criteria:

```text
.pte file exists
runner can invoke it
logits numerically compare against PyTorch within a rough tolerance
```

### Spike C: Export seq_len=512

Goal:

```text
repeat export with context=512
```

Exit criteria:

```text
no graph break that scales with sequence length
peak RSS measured
```

### Spike D: Greedy Decode

Goal:

```text
generate 32 tokens through ExecuTorch runner
```

Exit criteria:

```text
printed decoded text
TTFT and tokens/sec measured
```

### Spike E: Quantized Artifact

Goal:

```text
create Q8 and Q4 artifacts, or document exactly where quantization blocks
```

Exit criteria:

```text
artifact size table
peak RSS table
32-token decode still works
```

### Spike F: iOS Simulator Shell

Goal:

```text
run the artifact in an iOS-compatible app shell on Mac
```

Exit criteria:

```text
app loads tokenizer metadata and model artifact
generates at least one token
```

## Known Risks

### Remote Code Export

The checkpoint inherits custom Laguna model code. `torch.export` may fail on dynamic control flow, custom cache classes, or attention implementation branches.

Mitigation:

```text
export only a narrow fixed-shape forward first
replace generation/cache objects with tensors
fall back to a minimal local model wrapper if HF remote code is too dynamic
```

### KV Cache

A recompute-only loop can prove model portability but will not prove useful latency.

Mitigation:

```text
separate "can run" from "can run fast"
add explicit tensor KV cache after fixed-forward export lands
keep context=512 for first mobile target
```

### Low-Bit Quantization

Q4 may not be plug-and-play for all model operations.

Mitigation:

```text
try Q8 first
quantize linear weights before attention/cache complexity
keep llama.cpp scout alive as a faster quantization fallback
```

### Weight Duplication

Mobile failure often comes from loading duplicates, not just raw model size.

Mitigation:

```text
measure peak RSS during load and after warmup
explicitly fail if peak RSS is much larger than artifact size + KV + runtime overhead
```

## Deliverables

```text
scripts/export_executorch_dense.py
scripts/run_executorch_dense_smoke.py or C++ runner equivalent
runs/mobile_serving/executorch_export_<date>/
docs/runbooks/mobile-executorch.md
```

## Success Criteria

Minimum:

```text
fixed-shape forward exported and runnable
32-token decode works under native Mac smoke
```

Credible:

```text
Q4/Q8 artifact produced
simulated-phone profile under 3.5 GB peak RSS
TTFT and tokens/sec reported
```

Strong:

```text
iOS Simulator app shell runs the model artifact
same artifact has a clear path to real device testing
```
