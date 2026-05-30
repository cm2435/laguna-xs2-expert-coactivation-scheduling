from __future__ import annotations

import argparse
from pathlib import Path

from densify.rollout_sft.build_dataset import build_sft_rows, build_sft_rows_from_manifest, write_sft_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sandboxes-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-quality", action="append", default=["silver", "bronze"])
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--require-patch", action="store_true")
    parser.add_argument("--max-patch-bytes", type=int)
    parser.add_argument("--max-patch-lines", type=int)
    parser.add_argument("--turns-after-first-edit", type=int)
    parser.add_argument("--require-harness-success", action="store_true")
    parser.add_argument("--exclude-source-contains")
    args = parser.parse_args()

    if bool(args.runs_dir) == bool(args.manifest):
        parser.error("provide exactly one of --runs-dir or --manifest")
    if args.manifest:
        rows = build_sft_rows_from_manifest(
            args.manifest,
            include_qualities=set(args.include_quality),
            max_turns=args.max_turns,
            require_patch=args.require_patch,
            require_harness_success=args.require_harness_success,
            max_patch_bytes=args.max_patch_bytes,
            max_patch_lines=args.max_patch_lines,
            turns_after_first_edit=args.turns_after_first_edit,
            exclude_source_contains=args.exclude_source_contains,
        )
    else:
        rows = build_sft_rows(
            args.runs_dir,
            args.sandboxes_dir,
            include_qualities=set(args.include_quality),
            max_turns=args.max_turns,
            require_patch=args.require_patch,
            max_patch_bytes=args.max_patch_bytes,
            max_patch_lines=args.max_patch_lines,
            turns_after_first_edit=args.turns_after_first_edit,
        )
    write_sft_jsonl(rows, args.output)
    print(f"wrote_rows={len(rows)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
