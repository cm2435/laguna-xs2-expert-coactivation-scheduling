# Mobile Serving RFCs

**Status:** Active planning.

This folder owns the plan for taking the dense Laguna XS.2 checkpoint from "runs on GPU" toward an on-device serving demo. The immediate goal is not to prove full iPhone deployment on day one. The goal is to build a credible mobile-shaped serving path with phone-like memory constraints, while preserving a route to real iOS/Android deployment.

## Executive Summary

Our dense checkpoint is currently a custom Laguna-derived architecture, approximately 5.6 GB in BF16. That is too large for comfortable deployment on 6 GB phones such as iPhone 12 Pro Max or iPhone 14 Pro Max. A plausible mobile demo requires quantization, a short context window, no batching, careful KV limits, and a runtime that does not duplicate weights during load.

The recommended plan is:

1. Use **ExecuTorch** as the primary serving path because it is PyTorch-first and most likely to tolerate our custom dense architecture.
2. Use a **MacBook Pro simulated-phone profile** as the first target: same artifact, short context, strict memory budget, no batching, and iOS-compatible app/runtime where possible.
3. Run **llama.cpp/GGUF** and **MLC-LLM** as scouts. They may produce a faster demo if the architecture port is easy, but they should not block the primary path.
4. Treat real 6 GB iPhones as a stress target, not the initial success criterion. A newer 8-12 GB device is the more realistic practical mobile target.

## Current Model Facts

Target checkpoint:

```text
HF repo: cm2435-new/laguna-xs2-dense-k8-recon-2k
Commit: 25516298767404ce174ed0f3641e375c9ba4e060
Local upload folder: runs/hf_upload/laguna-xs2-dense-k8-recon-2k/
Weights: model.safetensors, approximately 5.6 GB BF16
```

Architecture facts:

```text
hidden_size: 2048
num_hidden_layers: 40
num_attention_heads: 48
num_key_value_heads: 8
head_dim: 128
vocab_size: 100352
attention: 30 sliding-window layers, 10 full-attention layers
sliding_window: 512
max_position_embeddings: 262144
```

Important caveat: this checkpoint is a serving target for infra work, not a high-quality coding model yet. The latest smoke eval reloads and generates, but coding quality is not solved.

## RFC Index

- [001: Mobile Targets And Memory Budget](001-mobile-targets-and-memory-budget.md)
- [002: ExecuTorch Serving Plan](002-executorch-serving-plan.md)
- [003: Simulated Phone On Mac Plan](003-simulated-phone-on-mac-plan.md)
- [004: Runtime Scout Plan](004-runtime-scout-plan.md)

## Working Definitions

```text
native Mac smoke:
    run the model locally without pretending to be a phone

simulated phone:
    run the same model artifact with phone-like constraints:
    short context, no batching, q4/q3 target, strict RSS budget

iOS Simulator:
    useful for app integration and iOS packaging;
    not a reliable proof of real iPhone memory behavior

real phone:
    final deployment target with actual iOS/Android memory pressure,
    thermal constraints, and app sandbox limits
```

## Near-Term Success Criterion

The first credible mobile-serving milestone is:

```text
An exported or runtime-loadable dense checkpoint runs a 32-token greedy decode
under a phone-shaped profile on Mac:

- context <= 512 tokens
- batch size = 1
- peak RSS below 3.5 GB
- no hidden 2x weight duplication after load
- tokens/sec and TTFT measured
- artifact path documented for iOS/Android next steps
```
