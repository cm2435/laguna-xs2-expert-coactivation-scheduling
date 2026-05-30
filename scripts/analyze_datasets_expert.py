"""
Measure Laguna-XS.2 MoE expert activation across the densification training
datasets (+ SWE-bench), one forward pass per "question"/prompt. Mirrors the C4
analysis (expert_activation_analysis.py): hooks every LagunaTopKRouter and
accumulates per-layer expert counts + coactivation, then dumps per-dataset
stats and sample questions.

Out: dataset_diag/<label>.npz, dataset_diag/<label>.json, dataset_diag/samples.json
"""
import json, os, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "poolside/Laguna-XS.2"
NUM_EXPERTS, TOP_K, SEQ_LEN = 256, 8, 1024
MAX_DOCS = 300
OUT = "/home/ubuntu/dataset_diag"
os.makedirs(OUT, exist_ok=True)

# label, hf_name, config, split, field, kind(language note)
DATASETS = [
    ("opencodeinstruct", "nvidia/OpenCodeInstruct", None, "train", "input", "Python (NL instruction)"),
    ("magicoder",        "ise-uiuc/Magicoder-Evol-Instruct-110K", None, "train", "instruction", "Multi-lang (NL instruction)"),
    ("codefeedback",     "m-a-p/CodeFeedback-Filtered-Instruction", None, "train", "query", "Multi-lang (NL instruction)"),
    ("kernelbook",       "GPUMODE/KernelBook", None, "train", "python_code", "Triton anchor (PyTorch module source)"),
    ("cuda_kernels",     "andrew-wang/cuda_kernels", None, "train", "kernel", "CUDA C++ (kernel source)"),
    ("swebench_lite",    "princeton-nlp/SWE-bench_Lite", None, "test", "problem_statement", "Repo issues (NL problem statement)"),
]

print(f"[load] {MODEL_ID}", flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
model.eval()
print(f"[load] done {time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

router_layers = []
for i, layer in enumerate(model.model.layers):
    gate = getattr(layer.mlp, "gate", None)
    if gate is not None and gate.__class__.__name__ == "LagunaTopKRouter":
        router_layers.append((i, gate))
layer_idxs = [i for i, _ in router_layers]
dev = next(model.parameters()).device
print(f"[hooks] {len(router_layers)} sparse-MoE routers", flush=True)

counts = {i: torch.zeros(NUM_EXPERTS, dtype=torch.float64, device=dev) for i in layer_idxs}
coact = {i: torch.zeros(NUM_EXPERTS, NUM_EXPERTS, dtype=torch.float64, device=dev) for i in layer_idxs}

def make_hook(li):
    def hook(module, inputs, output):
        sel = output[2]
        if sel.dim() == 1:
            sel = sel.unsqueeze(0)
        T = sel.shape[0]
        oh = torch.zeros(T, NUM_EXPERTS, device=sel.device, dtype=torch.float32)
        oh.scatter_(1, sel.long(), 1.0)
        counts[li] += oh.sum(0).double()
        coact[li] += (oh.t() @ oh).double()
    return hook

handles = [r.register_forward_hook(make_hook(i)) for i, r in router_layers]

def gini(x):
    x = np.sort(x.astype(np.float64)); n = len(x)
    if x.sum() == 0: return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n

def reset():
    for i in layer_idxs:
        counts[i].zero_(); coact[i].zero_()

def summarize(label, hf_name, kind, n_docs, total_tokens):
    per_layer = {}; union = np.zeros(NUM_EXPERTS); np_counts = {}; np_coact = {}
    for i in layer_idxs:
        c = counts[i].cpu().numpy(); np_counts[i] = c; np_coact[i] = coact[i].cpu().numpy()
        union += c
        p = c / c.sum() if c.sum() > 0 else c
        nz = p[p > 0]; ent = float(-(nz * np.log(nz)).sum())
        order = np.argsort(c)[::-1]
        per_layer[str(i)] = {
            "experts_used": int((c > 0).sum()), "coverage": float((c > 0).sum() / NUM_EXPERTS),
            "max_expert_share": float(c.max() / c.sum()) if c.sum() else 0.0,
            "gini_load": float(gini(c)), "normalized_entropy": float(ent / np.log(NUM_EXPERTS)),
            "effective_experts": float(np.exp(ent)),
            "top5_experts": [int(x) for x in order[:5]], "dead_experts": int((c == 0).sum())}
    gused = int((union > 0).sum())
    summary = {
        "label": label, "dataset": hf_name, "kind": kind,
        "config": {"num_experts": NUM_EXPERTS, "top_k": TOP_K, "sparse_layers": len(layer_idxs), "seq_len": SEQ_LEN},
        "corpus": {"docs": n_docs, "total_tokens": int(total_tokens)},
        "global": {
            "experts_ever_used": gused, "global_coverage": gused / NUM_EXPERTS,
            "mean_coverage_per_layer": float(np.mean([per_layer[str(i)]["coverage"] for i in layer_idxs])),
            "mean_gini_per_layer": float(np.mean([per_layer[str(i)]["gini_load"] for i in layer_idxs])),
            "mean_effective_experts": float(np.mean([per_layer[str(i)]["effective_experts"] for i in layer_idxs])),
            "mean_normalized_entropy": float(np.mean([per_layer[str(i)]["normalized_entropy"] for i in layer_idxs])),
            "global_top_experts": [int(x) for x in np.argsort(union)[::-1][:15]],
            "global_top_loads": [float(union[x]) for x in np.argsort(union)[::-1][:15]],
            "global_expert_load": union.tolist()},
        "per_layer": per_layer}
    np.savez_compressed(f"{OUT}/{label}.npz", layer_idxs=np.array(layer_idxs),
                        counts=np.stack([np_counts[i] for i in layer_idxs]),
                        coact=np.stack([np_coact[i] for i in layer_idxs]))
    json.dump(summary, open(f"{OUT}/{label}.json", "w"), indent=2)
    return summary

samples = {}
for label, hf_name, cfg, split, field, kind in DATASETS:
    print(f"\n[ds] {label} <- {hf_name}", flush=True)
    reset()
    try:
        ds = load_dataset(hf_name, cfg, split=split, streaming=True)
    except Exception as e:
        print(f"  SKIP load error: {repr(e)[:160]}", flush=True); continue
    texts, n, total = [], 0, 0
    t1 = time.time()
    with torch.inference_mode():
        for row in ds:
            q = row.get(field)
            if not q or not isinstance(q, str) or len(q.strip()) < 8:
                continue
            if len(texts) < 6:
                texts.append(q.strip())
            ids = tok(q, return_tensors="pt", truncation=True, max_length=SEQ_LEN).input_ids.to(dev)
            if ids.shape[1] < 2:
                continue
            model(ids, use_cache=False)
            total += ids.shape[1]; n += 1
            if n % 50 == 0:
                print(f"  {n}/{MAX_DOCS} q, {total} tok, {time.time()-t1:.0f}s", flush=True)
            if n >= MAX_DOCS:
                break
    s = summarize(label, hf_name, kind, n, total)
    samples[label] = {"dataset": hf_name, "kind": kind, "field": field,
                      "n_docs": n, "total_tokens": total,
                      "questions": [t[:500] for t in texts]}
    g = s["global"]
    print(f"  DONE {n} q / {total} tok | cover {100*g['global_coverage']:.1f}% "
          f"| eff {g['mean_effective_experts']:.1f} | gini {g['mean_gini_per_layer']:.3f}", flush=True)

for h in handles:
    h.remove()
json.dump(samples, open(f"{OUT}/samples.json", "w"), indent=2)
print("\n[done] wrote per-dataset stats + samples.json to", OUT, flush=True)
