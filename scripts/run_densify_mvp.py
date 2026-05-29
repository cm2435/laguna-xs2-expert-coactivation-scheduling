"""MVP: densify Laguna-XS.2 sparse MoE layers into dense FFNs and measure how
well each expert-scoring method reconstructs the true MoE block on held-out C4.

Reproduces the central claim of arXiv:2605.28207 on Laguna: expert *scoring*
dominates, and diversity-aware DO-ACP selection beats frequency/ACP. No
distillation -- this is the pre-distillation reconstruction signal.

Usage:
  python3 scripts/run_densify_mvp.py \
      --c4 /home/ubuntu/c4_excerpt.jsonl --layers 4 15 26 --ks 8 16 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densify import densify_layer as dl  # noqa: E402

MODEL_ID = "poolside/Laguna-XS.2"


def capture(model, tok, texts, layers, seq_len, device):
    """Forward C4 docs; capture per-layer MoE-block input X and true output Y."""
    store = {i: {"x": [], "y": []} for i in layers}
    handles = []

    def mk(i):
        def hook(_m, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            y = out.detach().reshape(-1, out.shape[-1])
            store[i]["x"].append(x.to("cpu"))
            store[i]["y"].append(y.to("cpu"))
        return hook

    for i in layers:
        handles.append(model.model.layers[i].mlp.register_forward_hook(mk(i)))

    ntok = 0
    with torch.inference_mode():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            model(ids, use_cache=False)
            ntok += ids.shape[1]
    for h in handles:
        h.remove()
    out = {}
    for i in layers:
        out[i] = (torch.cat(store[i]["x"]), torch.cat(store[i]["y"]))
    return out, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4", default="/home/ubuntu/c4_excerpt.jsonl")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 15, 26])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--calib-docs", type=int, default=120)
    ap.add_argument("--eval-docs", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--out", default="/home/ubuntu/densify_mvp_results.json")
    args = ap.parse_args()

    print(f"[load] {MODEL_ID}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    dev = next(model.parameters()).device
    print(f"[load] {time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    docs = [json.loads(l)["text"] for l in open(args.c4)]
    calib_docs = docs[:args.calib_docs]
    eval_docs = docs[args.calib_docs:args.calib_docs + args.eval_docs]
    print(f"[data] calib {len(calib_docs)} docs, eval {len(eval_docs)} docs", flush=True)

    print("[capture] calibration activations ...", flush=True)
    calib, ncal = capture(model, tok, calib_docs, args.layers, args.seq_len, dev)
    print("[capture] eval activations ...", flush=True)
    evalA, nev = capture(model, tok, eval_docs, args.layers, args.seq_len, dev)
    print(f"[capture] calib tokens {ncal}, eval tokens {nev}", flush=True)

    methods = ["frequency", "acp", "do-acp"]
    results = {"model": MODEL_ID,
               "config": {"layers": args.layers, "ks": args.ks,
                          "calib_tokens": ncal, "eval_tokens": nev,
                          "num_experts": int(model.config.num_experts),
                          "top_k": int(model.config.num_experts_per_tok)},
               "layers": {}}

    for i in args.layers:
        mlp = model.model.layers[i].mlp
        Xc, _ = calib[i]
        Xe, Ye = evalA[i]
        Xc, Xe, Ye = Xc.to(dev), Xe.to(dev), Ye.to(dev)
        print(f"\n[layer {i}] routing+expert stats on {Xc.shape[0]} calib tokens", flush=True)
        ts = time.time()
        routing = dl.compute_routing_stats(mlp, Xc)
        estats = dl.compute_expert_stats(mlp, Xc)
        print(f"  stats {time.time()-ts:.1f}s", flush=True)

        layer_res = {}
        for method in methods:
            for k in args.ks:
                if method == "frequency":
                    idx = dl.select_frequency(routing, k)
                elif method == "acp":
                    idx = dl.select_acp(routing, estats, k)
                else:
                    idx = dl.select_do_acp(routing, estats, k)
                eff = dl.effective_rank(estats, idx)
                dense = dl.build_dense_ffn(mlp, idx, routing, scaling="marginal")
                yp = dl.dense_forward(mlp, dense, Xe)
                rec = dl.reconstruction_error(Ye, yp)
                key = f"{method}_K{k}"
                layer_res[key] = {"method": method, "k": k,
                                  "rel_l2": rec["rel_l2"], "cosine": rec["cosine"],
                                  "effective_rank": eff,
                                  "experts": idx if k <= 16 else idx[:16] + ["..."]}
                print(f"  {key:16s} rel_l2={rec['rel_l2']:.4f} "
                      f"cos={rec['cosine']:.4f} eff_rank={eff:.1f}", flush=True)
        results["layers"][str(i)] = layer_res

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
