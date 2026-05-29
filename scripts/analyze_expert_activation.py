"""
Measure Laguna-XS.2 MoE expert activation on a C4 excerpt.

Hooks every LagunaTopKRouter (39 of 40 layers are sparse, top-8 of 256 experts),
runs C4 text through the model (forward only, batch=1 so no pad tokens pollute
stats), and accumulates per-layer:
  - expert token counts  [256]
  - coactivation matrix  [256, 256]  (experts co-selected within a token's top-8)
  - total routed tokens

Outputs: expert_stats.npz (raw) + expert_stats_summary.json (metrics).
"""
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "poolside/Laguna-XS.2"
C4_PATH = "/home/ubuntu/c4_excerpt.jsonl"
NUM_EXPERTS = 256
TOP_K = 8
SEQ_LEN = 1024
MAX_DOCS = 400

print(f"[load] {MODEL_ID}", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print(f"[load] done {time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ---- locate sparse-MoE routers, one per decoder layer that has one ----
layers = model.model.layers
router_layers = []  # (layer_idx, router_module)
for i, layer in enumerate(layers):
    mlp = layer.mlp
    gate = getattr(mlp, "gate", None)
    if gate is not None and gate.__class__.__name__ == "LagunaTopKRouter":
        router_layers.append((i, gate))
print(f"[hooks] {len(router_layers)} sparse-MoE routers found", flush=True)

dev = next(model.parameters()).device
counts = {i: torch.zeros(NUM_EXPERTS, dtype=torch.float64, device=dev) for i, _ in router_layers}
coact = {i: torch.zeros(NUM_EXPERTS, NUM_EXPERTS, dtype=torch.float64, device=dev) for i, _ in router_layers}
tokens_seen = {i: 0 for i, _ in router_layers}


def make_hook(layer_idx):
    def hook(module, inputs, output):
        # output = (router_logits, routing_weights, selected_experts)
        sel = output[2]  # [num_tokens, top_k]
        if sel.dim() == 1:
            sel = sel.unsqueeze(0)
        T = sel.shape[0]
        oh = torch.zeros(T, NUM_EXPERTS, device=sel.device, dtype=torch.float32)
        oh.scatter_(1, sel.long(), 1.0)  # binary membership per token
        counts[layer_idx] += oh.sum(0).double()
        coact[layer_idx] += (oh.t() @ oh).double()
        tokens_seen[layer_idx] += T
    return hook


handles = [r.register_forward_hook(make_hook(i)) for i, r in router_layers]

# ---- stream C4 excerpt, forward pass per doc ----
docs = []
with open(C4_PATH) as f:
    for line in f:
        docs.append(json.loads(line)["text"])
        if len(docs) >= MAX_DOCS:
            break
print(f"[data] {len(docs)} C4 docs", flush=True)

t1 = time.time()
total_tokens = 0
with torch.inference_mode():
    for n, text in enumerate(docs):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN).input_ids.to(dev)
        if ids.shape[1] < 2:
            continue
        model(ids, use_cache=False)
        total_tokens += ids.shape[1]
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(docs)} docs, {total_tokens} tokens, "
                  f"{(time.time()-t1):.1f}s", flush=True)
for h in handles:
    h.remove()
print(f"[run] {total_tokens} tokens in {time.time()-t1:.1f}s", flush=True)

# ---- metrics ----
def gini(x):
    x = np.sort(x.astype(np.float64))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n


layer_idxs = [i for i, _ in router_layers]
per_layer = {}
union_used = np.zeros(NUM_EXPERTS, dtype=np.float64)
np_counts = {}
np_coact = {}
for i in layer_idxs:
    c = counts[i].cpu().numpy()
    np_counts[i] = c
    np_coact[i] = coact[i].cpu().numpy()
    union_used += c
    toks = tokens_seen[i]
    used = int((c > 0).sum())
    p = c / c.sum() if c.sum() > 0 else c
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    norm_entropy = entropy / np.log(NUM_EXPERTS)
    # effective number of experts (perplexity of load distribution)
    eff_experts = float(np.exp(entropy))
    order = np.argsort(c)[::-1]
    per_layer[i] = {
        "tokens": int(toks),
        "experts_used": used,
        "coverage": used / NUM_EXPERTS,
        "mean_load_per_expert": float(c.mean()),
        "max_expert_share": float(c.max() / c.sum()) if c.sum() else 0.0,
        "gini_load": float(gini(c)),
        "normalized_entropy": float(norm_entropy),
        "effective_experts": eff_experts,
        "top5_experts": [int(x) for x in order[:5]],
        "top5_shares": [float(c[x] / c.sum()) for x in order[:5]],
        "dead_experts": int((c == 0).sum()),
    }

global_used = int((union_used > 0).sum())
summary = {
    "model": MODEL_ID,
    "config": {"num_experts": NUM_EXPERTS, "top_k": TOP_K,
               "sparse_layers": len(layer_idxs), "seq_len": SEQ_LEN},
    "corpus": {"source": "allenai/c4 (en) excerpt", "docs": len(docs),
               "total_tokens": int(total_tokens)},
    "global": {
        "experts_ever_used": global_used,
        "global_coverage": global_used / NUM_EXPERTS,
        "mean_coverage_per_layer": float(np.mean([per_layer[i]["coverage"] for i in layer_idxs])),
        "mean_gini_per_layer": float(np.mean([per_layer[i]["gini_load"] for i in layer_idxs])),
        "mean_effective_experts": float(np.mean([per_layer[i]["effective_experts"] for i in layer_idxs])),
        "mean_normalized_entropy": float(np.mean([per_layer[i]["normalized_entropy"] for i in layer_idxs])),
    },
    "per_layer": {str(i): per_layer[i] for i in layer_idxs},
}

np.savez_compressed(
    "/home/ubuntu/expert_stats.npz",
    layer_idxs=np.array(layer_idxs),
    counts=np.stack([np_counts[i] for i in layer_idxs]),
    coact=np.stack([np_coact[i] for i in layer_idxs]),
)
json.dump(summary, open("/home/ubuntu/expert_stats_summary.json", "w"), indent=2)

print("\n===== SUMMARY =====")
print(f"tokens: {total_tokens} | sparse layers: {len(layer_idxs)} | experts/layer: {NUM_EXPERTS} (top-{TOP_K})")
g = summary["global"]
print(f"experts ever used (any layer): {global_used}/{NUM_EXPERTS} ({100*g['global_coverage']:.1f}%)")
print(f"mean per-layer coverage: {100*g['mean_coverage_per_layer']:.1f}%")
print(f"mean effective experts/layer: {g['mean_effective_experts']:.1f} of {NUM_EXPERTS}")
print(f"mean load Gini: {g['mean_gini_per_layer']:.3f} | mean norm-entropy: {g['mean_normalized_entropy']:.3f}")
print("saved: expert_stats.npz, expert_stats_summary.json")
