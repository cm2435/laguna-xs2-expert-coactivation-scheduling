from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

PYTHON_REPOS = {
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", default="data/prompts/swebench_verified_python_tiny.jsonl")
    args = parser.parse_args()

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in dataset:
            repo = row.get("repo", "")
            if repo not in PYTHON_REPOS:
                continue
            prompt = (
                "You are fixing a Python repository issue. "
                "Write a concise diagnosis and the likely patch strategy.\n\n"
                f"Repository: {repo}\n"
                f"Issue:\n{row.get('problem_statement', '')}\n"
            )
            f.write(
                json.dumps(
                    {
                        "id": row["instance_id"],
                        "prompt": prompt,
                        "entrypoint": None,
                        "tests": [],
                    }
                )
                + "\n"
            )
            count += 1
            if count >= args.limit:
                break

    print(f"Wrote {count} prompts to {out}")


if __name__ == "__main__":
    main()
