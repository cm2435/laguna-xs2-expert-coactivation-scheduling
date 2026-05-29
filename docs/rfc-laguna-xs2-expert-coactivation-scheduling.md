# RFC: Expert Co-Activation Scheduling for Laguna XS.2

## Executive Summary

This project aims to improve **saturated decode throughput** for Laguna XS.2 under concurrent serving by scheduling requests that activate overlapping MoE experts into the same decode batches.

The core hypothesis is borrowed from Doubleword's "MoE expert co-activations" post: MoE inference is memory-bandwidth-bound because each batched forward may need to load many distinct expert weights. If similar prompts tend to route to similar experts, then batching similar prompts together reduces the number of unique expert loads per forward. Doubleword reported approximately 12-15% expert-load reduction with embedding-based ordering, and approximately 5.4% wall-clock improvement on Qwen3.5-35B-A3B, a close architectural cousin of Laguna XS.2.

For this hackathon, the concrete goal is:

> At fixed hardware and fixed concurrency, improve Laguna XS.2 saturated decode output tokens/sec by reducing unique routed-expert loads per decode batch.

This is **not** primarily a single-user latency optimization. It should be framed as a serving throughput and cost optimization. Tail latency may improve under high offered load if the server drains work faster, but reordering can also add batching delay, so p95 latency is a secondary metric rather than the lead claim.

## Background

Laguna XS.2 is a 33B-total / 3B-active MoE model designed for agentic coding and long-horizon work. Public model cards describe:

- 40 layers.
- 256 routed experts plus 1 shared expert.
- 3B activated parameters per token.
- Mixed attention: 30 sliding-window layers and 10 global-attention layers.
- FP8 KV cache.
- Available FP8, NVFP4, INT4, and BF16 variants.

The relevant property here is the large expert count. At decode time, each request contributes a small number of active experts per MoE layer. A heterogeneous batch can therefore touch a large union of experts, increasing HBM traffic and lowering throughput. A homogeneous batch should reuse more expert weights across tokens.

Doubleword's blog demonstrated this on Qwen/Qwen3.5-35B-A3B with top-8 routing over 256 experts and 40 MoE layers. Their oracle batching reduced expert loads by 21.3%; BGE embedding clustering reduced loads by 12.4%; a trained embedding model reduced loads by 15.6%; and the trained model produced a 5.4% wall-clock saving because MoE work was about 43% of the forward.

Sources:

- Doubleword: https://blog.doubleword.ai/moe-expert-coactivations
- Laguna XS.2 FP8 model card: https://huggingface.co/poolside/Laguna-XS.2-FP8
- Laguna XS.2 NVFP4 model card: https://huggingface.co/poolside/Laguna-XS.2-NVFP4

## Goal and Non-Goals

### Goal

Improve **saturated decode throughput** for Laguna XS.2 under concurrent serving.

Primary metric:

```text
decode output tokens/sec at fixed concurrency
```

Recommended concurrency sweep:

```text
C = 8, 16, 32, 64
```

Pick the highest concurrency that does not cause pathological queueing or OOM on the available machine, then use that as the main result.

### Mechanism Metric

Measure the number of unique routed experts used per decode step:

```text
for each decode step:
  for each MoE layer:
    count unique expert ids used by all tokens in the batch
sum over layers
```

Report:

```text
expert_load_reduction = 1 - clustered_unique_expert_loads / fcfs_unique_expert_loads
```

This is the key explanatory metric. If throughput moves, expert-load reduction explains why. If expert-load reduction moves but throughput does not, the project still produces a useful negative/diagnostic result about the actual bottleneck.

### Secondary Metrics

Latency should be measured, but not optimized first.

Report:

- p50 end-to-end latency.
- p95 end-to-end latency.
- p95 decode latency excluding queue wait, if available.
- Queue wait introduced by batching/reordering.
- TTFT only as a guardrail.

The expected behavior is:

- Unloaded single-request latency: little or no improvement.
- Saturated throughput: expected improvement.
- p95 latency under high offered load: may improve if throughput gain outweighs batching delay.
- TTFT: may worsen if we intentionally delay requests to form homogeneous batches.

### Non-Goals

- Do not claim a model-quality improvement.
- Do not claim single-user latency improvement unless measured.
- Do not change kernels in the first implementation.
- Do not require Blackwell-specific FP4 support.
- Do not start with multi-node expert-parallel serving.
- Do not make quantization the core contribution.

## Hardware Target

### Primary Target

Use a single GPU first:

- H100 80GB, or
- B200 if available, or
- A100 80GB if H100/B200 is not available.

Serve:

```text
poolside/Laguna-XS.2-FP8
```

Single-GPU serving is the right first target because:

- It is easier to interpret.
- It avoids expert-parallel sharding effects.
- Expert-load reduction maps directly to less expert-weight traffic inside one device.
- The scheduling idea is not Blackwell-specific.

### Stretch Target

Multi-GPU expert-parallel serving can be an appendix if time permits. It is harder because expert-load savings may vary by rank, and wall-clock speed is gated by the slowest rank. This is interesting but should not be the first implementation target.

## Data Plan

We need concurrent request traces that look like Laguna's real deployment target: agentic coding.

### Preferred Dataset

Use `pool` or a SWE-bench-style harness to generate agent trajectories.

Target:

```text
20 SWE-bench Verified tasks
multiple LLM calls per task
200-1000 agent turns total
```

For each model call, record:

- Request id.
- Trajectory id.
- Turn index.
- Prompt text.
- Output token count.
- Timing metadata.
- Routed experts per token/layer.
- Whether the turn is a prefill-heavy turn or a decode-heavy turn.

This supports the novel agentic question:

> Do turns in coding-agent trajectories remain coherent in expert space, and can scheduler policy exploit that coherence?

### Fallback Dataset

If `pool` is slow or hard to wire up, use a replay dataset of coding prompts:

- SWE-bench issue prompts.
- Repo-debugging prompts.
- Code-completion prompts.
- Patch-generation prompts.
- Tool-call / JSON / function-call prompts.
- File-inspection prompts from real agent traces if available.

Useful sources:

- SWE-bench Verified prompts.
- `mlabonne/open-perfectblend` for a broad reproduction-style baseline.
- WildChat or similar chat-only datasets for out-of-domain comparison.
- Locally captured turns from any coding-agent session.

### Minimum Viable Dataset

For the first spike:

```text
N = 256 prompts
max_new_tokens = 128 or 256
concurrency = 16 or 32
temperature = 0 or fixed low-temperature sampling
```

The first experiment only needs enough data to estimate whether expert-overlap headroom exists.

## Serving Stack Decision

### Use SGLang for the Hackathon MVP

SGLang has two practical surfaces that make this project much easier:

1. It can return routed experts.
2. It already has a `routing-key` scheduling policy.

Relevant local files inspected:

- `/Users/charliemasters/Desktop/synced_vm_002/hackathon/_deps/sglang/python/sglang/srt/state_capturer/routed_experts.py`
- `/Users/charliemasters/Desktop/synced_vm_002/hackathon/_deps/sglang/python/sglang/srt/managers/schedule_policy.py`
- `/Users/charliemasters/Desktop/synced_vm_002/hackathon/_deps/sglang/python/sglang/srt/entrypoints/openai/serving_base.py`

The important SGLang facts:

- `RoutedExpertsCapturer` captures top-k routed experts per layer.
- `--enable-return-routed-experts` enables routed-expert output.
- The OpenAI-compatible server can accept `return_routed_experts`.
- The server extracts a request routing key from the HTTP header:

```text
x-smg-routing-key
```

- `--schedule-policy routing-key` prioritizes waiting requests whose routing key is frequent in the running batch.

That means an MVP does not need a custom scheduler. We can:

1. Cluster requests outside the server.
2. Assign each request a cluster id.
3. Send the cluster id as `x-smg-routing-key`.
4. Let SGLang's existing scheduler preferentially keep same-key requests together.

### Use vLLM for Baseline/Profiling, Not First Implementation

vLLM is still useful:

- It has `enable_return_routed_experts=True`.
- It has a routed-experts example under `examples/rl/routed_experts_e2e.py`.
- It has EPLB load-stat logging for expert-parallel deployments.

Relevant local files inspected:

- `/Users/charliemasters/Desktop/synced_vm_002/hackathon/_deps/vllm/examples/rl/routed_experts_e2e.py`
- `/Users/charliemasters/Desktop/synced_vm_002/hackathon/_deps/vllm/vllm/v1/core/sched/request_queue.py`
- `/Users/charliemasters/Desktop/synced_vm_002/hackathon/_deps/vllm/docs/serving/expert_parallel_deployment.md`

However, vLLM's request queue path is less directly useful for the MVP: the V1 request queue exposes FCFS and priority scheduling, not a ready-made routing-key policy. Implementing co-activation scheduling in vLLM would be more invasive. Keep it as a comparison path or stretch.

## Experimental Conditions

Compare four conditions:

### 1. FCFS Baseline

Normal serving order.

This answers:

> What does the server do today?

### 2. Random Order Baseline

Shuffle requests before replay.

This answers:

> Is FCFS accidentally better or worse than random?

### 3. Embedding-Clustered Scheduling

Embed each prompt or agent turn with:

```text
BAAI/bge-small-en-v1.5
```

Cluster or order by cosine similarity, then assign routing keys:

```text
routing_key = cluster_id
```

Send requests with:

```text
x-smg-routing-key: cluster_07
```

This is the deployable version.

### 4. Oracle Expert-Overlap Scheduling

Use the observed routed experts to compute the best offline grouping.

Greedy batching algorithm:

1. Start a batch with one request.
2. Add the candidate request that increases the current batch's unique expert set the least.
3. Repeat until batch size is full.

This is not deployable because it uses future expert-routing information, but it estimates the headroom.

Report:

```text
embedding_captures = embedding_load_reduction / oracle_load_reduction
```

This matches Doubleword's framing.

## Implementation Plan

### Phase 0: Environment Setup

Clone and install SGLang with Laguna support.

Launch Laguna XS.2 FP8 with:

```bash
python -m sglang.launch_server \
  --model-path poolside/Laguna-XS.2-FP8 \
  --host 0.0.0.0 \
  --port 30000 \
  --schedule-policy fcfs \
  --enable-return-routed-experts \
  --tp 1
```

If SGLang requires a different CLI flag for tensor parallelism or model path in the installed version, adjust based on `python -m sglang.launch_server --help`.

Then launch the routing-key variant:

```bash
python -m sglang.launch_server \
  --model-path poolside/Laguna-XS.2-FP8 \
  --host 0.0.0.0 \
  --port 30000 \
  --schedule-policy routing-key \
  --enable-return-routed-experts \
  --tp 1
```

### Phase 1: Routed-Expert Capture Spike

Goal:

> Confirm we can get routed experts back from Laguna XS.2 through SGLang.

Send a small request:

```json
{
  "model": "poolside/Laguna-XS.2-FP8",
  "messages": [{"role": "user", "content": "Write a Python function to parse a unified diff."}],
  "max_tokens": 64,
  "temperature": 0,
  "return_routed_experts": true
}
```

Validate returned expert data:

- Shape is approximately `[tokens, layers, top_k]`.
- Expert ids are in `[0, 255]` for routed experts.
- Shared expert is not mixed into the routed ids unless SGLang reports fused shared experts separately.

Success criterion:

```text
We can save per-request routed experts to JSONL or Parquet.
```

Kill criterion:

```text
If routed experts cannot be returned for Laguna in SGLang within 2 hours, switch to vLLM profiling or monkey-patch SGLang's capturer.
```

### Phase 2: Offline Oracle Spike

Goal:

> Determine whether there is enough expert-overlap headroom to justify the project.

Run:

```text
N = 256 prompts
max_new_tokens = 128
temperature = 0
```

For each request, compute:

- Set of routed experts per layer across generated tokens.
- Aggregate expert set across all MoE layers.
- Pairwise Jaccard overlap between requests.

Then compare:

- Random batches.
- Greedy oracle batches.

Success criterion:

```text
oracle expert-load reduction >= 10%
```

Strong result:

```text
oracle expert-load reduction >= 15-20%
```

Kill criterion:

```text
oracle expert-load reduction < 5%
```

If oracle is weak, the scheduling idea probably will not land for this model/workload.

### Phase 3: Embedding Proxy Spike

Goal:

> See whether prompt embeddings can recover a useful fraction of oracle headroom.

Use:

```text
BAAI/bge-small-en-v1.5
```

Process:

1. Embed all prompts.
2. Cluster with k-means or greedy nearest-neighbor sorting.
3. Build batches by cluster.
4. Compute expert-load reduction using already-captured routed experts.

Success criterion:

```text
embedding expert-load reduction >= 5%
```

Strong result:

```text
embedding captures >= 50% of oracle reduction
```

If the oracle is strong but BGE is weak, consider a tiny trained proxy:

- Input: prompt text.
- Target: expert-overlap Jaccard similarity.
- Loss: contrastive or pairwise regression.
- Train on 1k-10k captured request pairs if time permits.

This is a stretch. The MVP should not depend on training a proxy.

### Phase 4: SGLang Routing-Key Serving Experiment

Goal:

> Convert offline expert-load reduction into measured throughput.

Conditions:

1. `--schedule-policy fcfs`, no routing key.
2. `--schedule-policy routing-key`, random routing key.
3. `--schedule-policy routing-key`, embedding cluster id as routing key.
4. Optional: oracle cluster id as routing key.

Replay requests concurrently with a fixed offered load.

Recommended settings:

```text
concurrency = 16, 32, 64
max_new_tokens = 128 or 256
temperature = 0
stream = false for simpler timing, true if measuring streaming latency
```

Request header:

```text
x-smg-routing-key: cluster_07
```

Measure:

- Requests/sec.
- Output tokens/sec.
- Mean output tokens/request.
- p50/p95 end-to-end latency.
- p50/p95 queue wait if available.
- GPU utilization.
- Expert-load reduction from returned routed experts.

Primary result:

```text
throughput_gain = clustered_decode_tokens_per_sec / fcfs_decode_tokens_per_sec - 1
```

Success criterion:

```text
throughput gain >= 3%
```

Strong result:

```text
throughput gain >= 5%
expert-load reduction >= 10%
```

### Phase 5: Agentic Trajectory Extension

Goal:

> Answer the novel question Doubleword left open: can ordering work continuously for in-flight agent sessions?

Data:

- Multiple coding-agent trajectories.
- Each trajectory contains ordered turns.

Analyses:

1. Consecutive-turn expert overlap:

```text
Jaccard(experts(turn_t), experts(turn_t+1))
```

2. Random-turn overlap:

```text
Jaccard(experts(turn_i), experts(turn_j)) for random i, j
```

3. Overlap decay:

```text
Jaccard(turn_t, turn_t+k), k = 1, 2, 4, 8
```

4. Cross-session scheduling:

At each scheduling point, group in-flight sessions by:

- same trajectory id,
- embedding cluster,
- previous-turn expert set, if available.

This gives the hackathon story a more original angle:

> Doubleword showed static prompt reordering. We test whether long-running coding agents have temporal expert coherence, and whether a continuous scheduler can exploit it.

## Analysis Scripts to Build

### `capture_requests.py`

Responsibilities:

- Load prompt JSONL.
- Send requests to SGLang.
- Include optional routing key header.
- Request routed experts.
- Save response, timings, and routed experts.

Output:

```text
runs/{run_id}/responses.jsonl
```

### `compute_expert_loads.py`

Responsibilities:

- Decode routed-expert blobs if needed.
- Compute unique experts per layer per request.
- Compute batch-level unique expert loads.
- Compute random, clustered, and oracle reductions.

Output:

```text
runs/{run_id}/expert_load_summary.csv
runs/{run_id}/expert_overlap_matrix.npy
```

### `cluster_prompts.py`

Responsibilities:

- Embed prompts with BGE.
- Produce cluster ids.
- Produce sorted replay order.

Output:

```text
runs/{run_id}/clustered_prompts.jsonl
```

### `bench_replay.py`

Responsibilities:

- Replay prompts at controlled concurrency.
- Attach `x-smg-routing-key`.
- Measure throughput and latency.

Output:

```text
runs/{run_id}/bench_summary.json
runs/{run_id}/latencies.csv
```

## Reporting Plan

The final writeup should include four charts:

1. Expert-load reduction by method:

```text
random / FCFS / BGE-clustered / oracle
```

2. Decode output tokens/sec by method at fixed concurrency.

3. p50/p95 latency under fixed offered load.

4. Agentic temporal coherence:

```text
expert-overlap vs turn distance
```

Suggested headline if results land:

> Expert-aware request scheduling improves Laguna XS.2 saturated decode throughput by X% at concurrency Y, driven by Z% fewer unique routed-expert loads per decode batch. Agentic coding turns show measurable temporal expert coherence, suggesting continuous co-activation scheduling is a natural fit for long-running coding agents.

## Risks and Gotchas

### This May Not Improve Single-Request Latency

This is a throughput/cost optimization. Do not sell it as "faster for one user" unless measured.

### Queueing Delay Can Hide the Win

If we delay requests too long to form homogeneous batches, p95 latency can worsen. Keep batching delay bounded, and report queue wait separately.

### Expert-Load Reduction May Not Become Wall-Clock Reduction

Doubleword saw 12.3% expert-load reduction become 5.4% wall-clock reduction. We should expect the wall-clock gain to be smaller than the mechanism metric.

### Prefix Caching Can Confound Agent Runs

Agentic workloads reuse long prefixes. Routed experts may only be captured for newly computed tokens, depending on serving stack behavior. Separate:

- prefill tokens,
- prefix-cache hits,
- decode tokens.

### Embeddings May Be a Weak Proxy

The oracle may show headroom while BGE fails to capture it. This is still informative, but less demo-friendly. Keep the oracle and BGE results separate.

### SGLang Routing-Key Policy Is Coarse

The built-in policy groups by exact routing key, not continuous expert-overlap score. This is good for an MVP, but a custom scheduler could do better.

### Multi-GPU Expert Parallelism Is Complicated

Expert-load savings may be uneven across ranks. The slowest rank gates wall-clock speed. Single-GPU first.

## Early Spike Checklist

Run these in order.

1. Confirm SGLang can serve `poolside/Laguna-XS.2-FP8` on the available GPU.
2. Confirm `return_routed_experts` works on one request.
3. Capture routed experts for 32 prompts.
4. Compute random vs oracle expert-load reduction.
5. Scale to 256 prompts if the first oracle result is promising.
6. Add BGE embeddings and compute clustered expert-load reduction.
7. Launch SGLang with `--schedule-policy routing-key`.
8. Replay at concurrency 16/32 with random keys vs BGE cluster keys.
9. Measure output tokens/sec and p95 latency.
10. Decide whether to extend to agent trajectories.

## Go / No-Go Criteria

Go if:

- Routed expert capture works.
- Oracle expert-load reduction is at least 10%.
- BGE or simple clustering captures at least 5% expert-load reduction.
- Routing-key serving produces at least 3% output-token throughput improvement.

Pivot if:

- Routed expert capture does not work after 2 hours.
- Oracle expert-load reduction is below 5%.
- Throughput does not move despite expert-load reduction, suggesting another bottleneck dominates.

Potential pivots:

- Expert pruning using the same activation logs.
- Observation-as-draft-source speculative decoding.
- Pure profiling/audit writeup: "Why co-activation scheduling does/does not transfer to Laguna XS.2."

## Open Questions

- Does Laguna XS.2 route coding-agent turns more coherently than generic chat prompts?
- Does prompt embedding similarity predict expert overlap well enough for deployment?
- Does previous-turn expert set predict next-turn expert set better than prompt embeddings?
- How much batching delay is acceptable before p95 latency worsens?
- Does the FP8 variant behave differently from NVFP4 in expert-routing stability?
- Does SGLang's routing-key policy produce enough co-location, or do we need a custom continuous scheduler?

