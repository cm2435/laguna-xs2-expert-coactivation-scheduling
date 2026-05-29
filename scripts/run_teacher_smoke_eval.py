from __future__ import annotations

import argparse
import json
from pathlib import Path

from densify.code_scoring import score_code_generation
from densify.config import load_teacher_smoke_config
from densify.generation import generate_one
from densify.model_introspection import architecture_summary
from densify.prompt_data import load_jsonl_prompts
from densify.run_artifacts import append_jsonl, new_run_dir, write_json, write_yaml
from densify.teacher_loader import load_teacher_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_teacher_smoke_config(args.config)
    prompts = load_jsonl_prompts(cfg.prompt_path)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    run_dir = new_run_dir(cfg.output_dir, prefix="teacher_smoke")
    write_yaml(run_dir / "config_resolved.yaml", cfg)

    tokenizer = load_tokenizer(cfg.model_id, cfg.trust_remote_code)
    model = load_teacher_model(
        cfg.model_id,
        torch_dtype=cfg.torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
        device_map=cfg.device_map,
        compressed_tensors_run_compressed=cfg.compressed_tensors_run_compressed,
    )
    architecture = architecture_summary(model, cfg.model_id, cfg.torch_dtype)
    write_json(run_dir / "architecture.json", architecture)

    generations_path = run_dir / "generations.jsonl"
    counters = {
        "num_prompts": 0,
        "non_empty": 0,
        "python_like": 0,
        "parse_ok": 0,
        "tests_ok": 0,
        "total_generated_tokens": 0,
        "total_latency_s": 0.0,
    }

    examples: list[str] = []
    for prompt in prompts:
        result = generate_one(model, tokenizer, prompt, cfg.generation)
        score = score_code_generation(result.text, prompt.tests)
        row = {
            "id": prompt.id,
            "prompt": prompt.prompt,
            "raw_generation": result.text,
            "extracted_code": score.extracted_code,
            "parse_ok": score.parse_ok,
            "tests_ok": score.tests_ok,
            "test_stdout": score.test_stdout,
            "test_stderr": score.test_stderr,
            "latency_s": result.latency_s,
            "generated_tokens": result.generated_tokens,
        }
        append_jsonl(generations_path, row)

        counters["num_prompts"] += 1
        counters["non_empty"] += int(bool(result.text.strip()))
        has_code_block = "```python" in result.text.lower()
        has_function = "def " in score.extracted_code
        counters["python_like"] += int(has_function or has_code_block)
        counters["parse_ok"] += int(score.parse_ok)
        counters["tests_ok"] += int(score.tests_ok)
        counters["total_generated_tokens"] += result.generated_tokens
        counters["total_latency_s"] += result.latency_s

        if len(examples) < 3:
            examples.append(
                f"## {prompt.id}\n\n"
                f"Prompt:\n{prompt.prompt}\n\n"
                f"Generation:\n```python\n{score.extracted_code}\n```\n"
            )

    n = max(counters["num_prompts"], 1)
    summary = {
        **counters,
        "non_empty_rate": counters["non_empty"] / n,
        "python_like_rate": counters["python_like"] / n,
        "parse_ok_rate": counters["parse_ok"] / n,
        "tests_ok_rate": counters["tests_ok"] / n,
        "tokens_per_second": counters["total_generated_tokens"]
        / max(counters["total_latency_s"], 1e-6),
    }
    write_json(run_dir / "summary.json", summary)
    Path(run_dir / "examples.md").write_text("\n".join(examples), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote={run_dir}")


if __name__ == "__main__":
    main()
