from __future__ import annotations

import argparse
import signal

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

FENCE = chr(96) * 3  # ``` without typing literal backticks


def parse_args():
    p = argparse.ArgumentParser(description="HumanEval pass@1 with per-problem output dump.")
    p.add_argument("--model", default="./sft_model")
    p.add_argument("--num", type=int, default=10, help="number of problems; -1 = all 164")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", default="humaneval_outputs.md")
    p.add_argument("--device", default="cuda")
    p.add_argument("--timeout", type=int, default=10)
    return p.parse_args()


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def extract_code(text: str) -> str:
    """Pull the function body out of a chat completion (handles ``` fences)."""
    if FENCE in text:
        block = text.split(FENCE)[1]
        if block.lower().startswith("python"):
            block = block[len("python"):]
        return block.strip()
    return text.strip()


def run_tests(code: str, test_src: str, entry_point: str, timeout: int):
    program = code + "\n\n" + test_src + "\n\n" + "check(" + entry_point + ")\n"
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        exec(program, {})
        return True, ""
    except Timeout:
        return False, "timeout"
    except Exception as exc:  # noqa: BLE001 - any failure = wrong
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": args.device}
    )
    model.eval()

    ds = load_dataset("openai/openai_humaneval", split="test")
    n = len(ds) if args.num < 0 else min(args.num, len(ds))

    passed = 0
    lines = ["# HumanEval outputs — " + args.model, ""]
    for i in range(n):
        row = ds[i]
        task_id = row["task_id"]
        entry = row["entry_point"]
        instruction = (
            "Complete this Python function. Return only the function in a code block.\n\n"
            + row["prompt"]
        )
        text = tok.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(args.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = extract_code(gen)
        ok, err = run_tests(code, row["test"], entry, args.timeout)
        passed += int(ok)

        verdict = "PASS" if ok else "FAIL"
        print(task_id + ": " + verdict + ("" if ok else "  (" + err + ")"), flush=True)
        lines += [
            "## " + task_id + " — " + verdict,
            "",
            "" if ok else ("error: `" + err + "`"),
            "",
            FENCE + "python",
            code,
            FENCE,
            "",
        ]

    summary = "pass@1: " + str(passed) + "/" + str(n) + " = " + str(round(100 * passed / n, 1)) + "%"
    print(summary, flush=True)
    lines.insert(1, "**" + summary + "**")
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
