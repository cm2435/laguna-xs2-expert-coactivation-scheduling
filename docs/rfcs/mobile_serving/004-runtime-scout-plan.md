# RFC 004: Runtime Scout Plan

**Status:** Active planning.

## Purpose

Define the parallel scout work for runtimes that might beat ExecuTorch on demo velocity or performance, without letting them derail the primary custom-architecture path.

The primary path remains ExecuTorch. Scouts are time-boxed checks for faster roads.

## Runtime Ranking

```text
primary: ExecuTorch
scout 1: llama.cpp / GGUF
scout 2: MLC-LLM
lower priority: ONNX Runtime Mobile, MNN, MediaPipe
```

## Scout 1: llama.cpp / GGUF

### Why Scout It

llama.cpp has the best practical mobile story for quantized LLM demos:

```text
mmap-first model loading
excellent low-bit quantization formats
CPU and Metal backends
iOS/Android community paths
simple command-line measurement story
```

If our dense architecture can be represented as an existing Llama-like variant with minor changes, this could be the fastest route to a demo.

### What To Check

```text
can the dense checkpoint be converted to GGUF?
do the tensor names and shapes map to an existing architecture?
does attention support grouped-query heads with head_dim=128 and 48/8 head split?
can the 30 sliding-window + 10 full-attention pattern be represented?
does tokenizer conversion preserve Laguna tokens?
```

### Kill Criteria

Stop the scout if any of these are true:

```text
requires a deep new llama.cpp architecture port
requires rewriting attention/KV cache internals
cannot represent mixed sliding/full attention without invasive changes
tokenizer conversion blocks for more than a few hours
```

### Success Criteria

```text
GGUF artifact created
q4 artifact loads
32-token generation works on Mac
peak RSS and tokens/sec measured
```

## Scout 2: MLC-LLM

### Why Scout It

MLC-LLM is strong for mobile and browser-ish deployment:

```text
iOS app path
TVM compilation and Metal support
good deployment-oriented runtime controls
```

It may be a good demo path if model support can be added cleanly.

### What To Check

```text
can the model config be expressed in MLC's model registry?
how much code is required for Laguna-style attention?
can we compile a one-layer or tiny dense model first?
does the tokenizer path work?
does generated artifact size look plausible?
```

### Kill Criteria

Stop the scout if:

```text
custom architecture support requires broad compiler/runtime changes
one-layer compile cannot land quickly
attention pattern requires unsupported lowering
```

### Success Criteria

```text
tiny architecture compiles
full checkpoint compile path is understood
iOS or Mac runner can be started with a toy artifact
```

## Lower-Priority Runtimes

### ONNX Runtime Mobile

Potentially useful for fixed graphs, but LLM generation and custom KV cache handling can become awkward. Consider only if ExecuTorch export fails for reasons that ONNX avoids.

### MNN

Mobile-friendly, but less obvious for our custom PyTorch-first checkpoint and hackathon timeline.

### MediaPipe

Good for app pipelines, not the right first choice for custom LLM serving.

## Suggested Ownership

```text
owner A: ExecuTorch export path
owner B: simulated-phone measurement harness
owner C: llama.cpp GGUF scout
owner D: MLC-LLM scout, if team capacity allows
```

## Scout Reporting Template

Each scout should produce:

```text
runtime:
commit/version:
checkpoint tested:
conversion status:
generation status:
memory status:
main blocker:
estimated remaining work:
recommendation: continue / pause / abandon
```

## Decision Rule

After the first scout window, pick the path with the best combination of:

```text
loads the real checkpoint
supports q4 or better
measures peak RSS honestly
can become an iOS/Android demo
does not require deep runtime surgery
```

If no scout beats ExecuTorch within the time box, continue with ExecuTorch.
