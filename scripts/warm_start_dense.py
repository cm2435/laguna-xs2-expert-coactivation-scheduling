"""DO-ACP warm-start: initialize the student's dense routed_dense FFNs by
concatenating the K diversity-optimal experts (Gram log-det selection) from the
teacher MoE, instead of random init. Implements KRAFTON's score->select->concat
(arXiv:2605.28207) as the Stage-0 init for reconstruction pretraining.

Saves a warm-started student checkpoint to --output-dir.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from densify import densify_layer as dl  # noqa: E402
from densify.reconstruction import find_reconstruction_layer_ids  # noqa: E402
from densify.reconstruction_data import format_sft_row  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="poolside/Laguna-XS.2")
    ap.add_argument("--student-model", default="cm2435-new/laguna-xs2-dense-k8-copied-shell")
    ap.add_argument("--dataset", default="nvidia/OpenCodeInstruct")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8, help="experts to select (K*512 must = dense width)")
    ap.add_argument("--calib-rows", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--scaling", default="marginal", choices=["marginal", "uniform"])
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("[load] teacher + student", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"})
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda"})
    teacher.eval()
    dev = next(teacher.parameters()).device

    layer_ids = find_reconstruction_layer_ids(teacher, student)
    print(f"[layers] {len(layer_ids)} sparse layers to warm-start", flush=True)

    # ---- calibration: capture teacher MoE-block inputs per layer ----
    rows = load_dataset(args.dataset, split="train", streaming=True)
    texts = []
    for r in rows:
        try:
            texts.append(format_sft_row(r))
        except ValueError:
            continue
        if len(texts) >= args.calib_rows:
            break
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in layer_ids}
    handles = []

    def mk(i):
        def hook(_m, inp, _out):
            captured[i].append(inp[0].detach().reshape(-1, inp[0].shape[-1]))
        return hook

    teacher_layers = teacher.model.layers
    for i in layer_ids:
        handles.append(teacher_layers[i].mlp.register_forward_hook(mk(i)))
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=args.seq_len).input_ids.to(dev)
            if ids.shape[1] < 2:
                continue
            teacher(ids, use_cache=False)
    for h in handles:
        h.remove()

    # ---- per-layer DO-ACP select + concat into student.routed_dense ----
    student_layers = student.model.layers
    t0 = time.time()
    for n, i in enumerate(layer_ids):
        X = torch.cat(captured[i], dim=0)
        tmlp = teacher_layers[i].mlp
        routing = dl.compute_routing_stats(tmlp, X)
        estats = dl.compute_expert_stats(tmlp, X)
        idx = dl.select_do_acp(routing, estats, args.k)
        dense = dl.build_dense_ffn(tmlp, idx, routing, scaling=args.scaling)
        rd = student_layers[i].mlp.routed_dense
        with torch.no_grad():
            rd.gate_proj.weight.copy_(dense["gate"].to(rd.gate_proj.weight.dtype))
            rd.up_proj.weight.copy_(dense["up"].to(rd.up_proj.weight.dtype))
            rd.down_proj.weight.copy_(dense["down"].to(rd.down_proj.weight.dtype))
        captured[i] = []  # free
        if (n + 1) % 5 == 0 or n == len(layer_ids) - 1:
            print(f"  warm-started {n+1}/{len(layer_ids)} layers, eff_rank(L{i})="
                  f"{dl.effective_rank(estats, idx):.1f}, {time.time()-t0:.0f}s", flush=True)

    print("[save] writing warm-started student", flush=True)
    student.save_pretrained(args.output_dir, safe_serialization=True)
    tok.save_pretrained(args.output_dir)
    print(f"[done] warm-started student -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
