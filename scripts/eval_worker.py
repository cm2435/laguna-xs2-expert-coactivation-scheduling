"""Isolated kernel evaluator — runs ONE kernel's compile+execution in its own
process so a faulty kernel (illegal memory access etc.) can't corrupt the CUDA
context of the model/eval driver. Reads {code, dsl, op} JSON from --in, writes
result JSON to --out."""
import os, sys, json, argparse
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REFS = {
    "relu": "torch.relu", "tanh": "torch.tanh", "sigmoid": "torch.sigmoid",
    "gelu": "lambda x: torch.nn.functional.gelu(x)", "abs": "torch.abs",
    "silu": "torch.nn.functional.silu", "softplus": "torch.nn.functional.softplus",
    "elu": "torch.nn.functional.elu",
    "leakyrelu": "lambda x: torch.nn.functional.leaky_relu(x, 0.01)",
    "mish": "torch.nn.functional.mish",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()
    job = json.load(open(a.inp))
    import torch  # noqa
    from densify.kernel_reward import evaluate_kernel, evaluate_triton
    ref = eval(REFS[job["op"]], {"torch": torch})
    if job["dsl"] == "CUDA":
        r = evaluate_kernel(job["code"], ref, name=job.get("name", job["op"]), timeout=75)
    else:
        r = evaluate_triton(job["code"], ref, timeout=60)
    json.dump(r, open(a.out, "w"))


if __name__ == "__main__":
    main()
