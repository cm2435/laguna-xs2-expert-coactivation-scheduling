from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from datasets import load_dataset


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_hf_rows(
    dataset: str,
    *,
    config: str | None,
    split: str,
    rows: int,
    seed: int,
    shuffle_buffer: int,
) -> list[dict]:
    dataset_kwargs = {"split": split, "streaming": True}
    if config:
        loaded = load_dataset(dataset, config, **dataset_kwargs)
    else:
        loaded = load_dataset(dataset, **dataset_kwargs)
    if shuffle_buffer > 0:
        loaded = loaded.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return [dict(row) for row in islice(loaded, rows)]


def limit_rows(rows: list[dict], max_rows: int | None) -> list[dict]:
    if max_rows is None:
        return rows
    return rows[:max_rows]


def repeat_rows(rows: list[dict], repeat: int) -> list[dict]:
    repeated: list[dict] = []
    for index in range(max(repeat, 0)):
        for row in rows:
            copy = dict(row)
            copy["_reconstruction_repeat"] = index
            repeated.append(copy)
    return repeated


def tag_rows(rows: Iterable[dict], source: str) -> list[dict]:
    tagged = []
    for row in rows:
        clean = dict(row)
        clean["_reconstruction_source"] = source
        tagged.append(clean)
    return tagged


def round_robin(groups: list[list[dict]], max_rows: int) -> list[dict]:
    output: list[dict] = []
    index = 0
    while len(output) < max_rows and any(index < len(group) for group in groups):
        for group in groups:
            if index < len(group):
                output.append(group[index])
                if len(output) >= max_rows:
                    break
        index += 1
    return output


def build_mixture(
    *,
    code_path: Path | None = None,
    code_dataset: str | None = None,
    code_dataset_config: str | None = None,
    code_split: str = "train",
    code_rows: int | None = None,
    code_shuffle_buffer: int = 10_000,
    tool_path: Path,
    tool_rows: int | None = None,
    tool_repeat: int = 1,
    recovery_path: Path,
    recovery_rows: int | None = None,
    recovery_repeat: int = 1,
    output_path: Path,
    max_rows: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    if not code_path and not code_dataset:
        raise ValueError("Either code_path or code_dataset is required")
    if code_path and code_dataset:
        raise ValueError("Use only one of code_path or code_dataset")

    if code_dataset:
        if code_rows is None:
            raise ValueError("code_rows is required when using code_dataset")
        code = read_hf_rows(
            code_dataset,
            config=code_dataset_config,
            split=code_split,
            rows=code_rows,
            seed=seed,
            shuffle_buffer=code_shuffle_buffer,
        )
    else:
        assert code_path is not None
        code = read_jsonl(code_path)

    groups = [
        tag_rows(limit_rows(code, code_rows), "code"),
        tag_rows(repeat_rows(limit_rows(read_jsonl(tool_path), tool_rows), tool_repeat), "tool"),
        tag_rows(
            repeat_rows(limit_rows(read_jsonl(recovery_path), recovery_rows), recovery_repeat),
            "recovery",
        ),
    ]
    for group in groups:
        rng.shuffle(group)
    rows = round_robin(groups, max_rows=max_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "rows_written": len(rows),
        "sources": {
            source: sum(row["_reconstruction_source"] == source for row in rows)
            for source in ["code", "tool", "recovery"]
        },
        "seed": seed,
        "code_dataset": code_dataset,
        "code_dataset_config": code_dataset_config,
        "code_split": code_split,
        "tool_repeat": tool_repeat,
        "recovery_repeat": recovery_repeat,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a mixed corpus for dense reconstruction/KL."
    )
    code_source = parser.add_mutually_exclusive_group(required=True)
    code_source.add_argument("--code-path", type=Path)
    code_source.add_argument("--code-dataset")
    parser.add_argument("--code-dataset-config")
    parser.add_argument("--code-split", default="train")
    parser.add_argument("--code-rows", type=int)
    parser.add_argument("--code-shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--tool-path", type=Path, required=True)
    parser.add_argument("--tool-rows", type=int)
    parser.add_argument("--tool-repeat", type=int, default=1)
    parser.add_argument("--recovery-path", type=Path, required=True)
    parser.add_argument("--recovery-rows", type=int)
    parser.add_argument("--recovery-repeat", type=int, default=1)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    print(json.dumps(build_mixture(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
