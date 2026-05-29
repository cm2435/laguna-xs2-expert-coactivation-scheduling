from __future__ import annotations

import json
from pathlib import Path

PROMPTS = [
    {
        "id": "sort_001",
        "prompt": (
            "Write a Python function sort_numbers(xs) that returns the numbers in ascending "
            "order."
        ),
        "entrypoint": "sort_numbers",
        "tests": [
            "assert sort_numbers([3, 1, 2]) == [1, 2, 3]",
            "assert sort_numbers([]) == []",
        ],
    },
    {
        "id": "binary_search_001",
        "prompt": (
            "Write a Python function binary_search(xs, target) that returns the index of "
            "target in sorted list xs, or -1 if target is absent."
        ),
        "entrypoint": "binary_search",
        "tests": [
            "assert binary_search([1, 3, 5, 7], 5) == 2",
            "assert binary_search([1, 3, 5, 7], 2) == -1",
        ],
    },
    {
        "id": "diff_paths_001",
        "prompt": (
            "Write a Python function changed_files(diff_text) that parses a unified diff "
            "string and returns a sorted list of changed file paths."
        ),
        "entrypoint": "changed_files",
        "tests": [
            "d = 'diff --git a/a.py b/a.py\\n--- a/a.py\\n+++ b/a.py\\n@@ -1 +1 @@\\n-x\\n+y\\n'",
            "assert changed_files(d) == ['a.py']",
        ],
    },
    {
        "id": "off_by_one_001",
        "prompt": (
            "Write a Python function count_items(xs) that returns the number of items in xs "
            "without using len()."
        ),
        "entrypoint": "count_items",
        "tests": [
            "assert count_items([]) == 0",
            "assert count_items(['a', 'b', 'c']) == 3",
        ],
    },
    {
        "id": "traceback_001",
        "prompt": (
            "Write a Python function safe_divide(a, b) that returns None when b is zero and "
            "otherwise returns a / b."
        ),
        "entrypoint": "safe_divide",
        "tests": [
            "assert safe_divide(6, 2) == 3",
            "assert safe_divide(6, 0) is None",
        ],
    },
    {
        "id": "dedupe_001",
        "prompt": (
            "Write a Python function unique_preserve_order(xs) that removes duplicates while "
            "preserving first occurrence order."
        ),
        "entrypoint": "unique_preserve_order",
        "tests": [
            "assert unique_preserve_order([1, 2, 1, 3, 2]) == [1, 2, 3]",
            "assert unique_preserve_order([]) == []",
        ],
    },
    {
        "id": "palindrome_001",
        "prompt": (
            "Write a Python function is_palindrome(text) that ignores case and "
            "non-alphanumeric characters."
        ),
        "entrypoint": "is_palindrome",
        "tests": [
            "assert is_palindrome('A man, a plan, a canal: Panama') is True",
            "assert is_palindrome('hello') is False",
        ],
    },
    {
        "id": "flatten_001",
        "prompt": (
            "Write a Python function flatten_once(items) that flattens one level of nested "
            "lists."
        ),
        "entrypoint": "flatten_once",
        "tests": [
            "assert flatten_once([[1, 2], [3], [], [4, 5]]) == [1, 2, 3, 4, 5]",
            "assert flatten_once([]) == []",
        ],
    },
    {
        "id": "chunk_001",
        "prompt": (
            "Write a Python function chunks(xs, size) that returns a list of lists split into "
            "chunks of length size."
        ),
        "entrypoint": "chunks",
        "tests": [
            "assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
            "assert chunks([], 3) == []",
        ],
    },
    {
        "id": "word_count_001",
        "prompt": (
            "Write a Python function word_counts(text) that returns a dict mapping lowercase "
            "words to counts."
        ),
        "entrypoint": "word_counts",
        "tests": [
            "assert word_counts('Hello hello world') == {'hello': 2, 'world': 1}",
            "assert word_counts('') == {}",
        ],
    },
    {
        "id": "merge_dicts_001",
        "prompt": (
            "Write a Python function merge_counts(a, b) that adds values for matching keys "
            "from two dictionaries."
        ),
        "entrypoint": "merge_counts",
        "tests": [
            "assert merge_counts({'a': 1}, {'a': 2, 'b': 3}) == {'a': 3, 'b': 3}",
            "assert merge_counts({}, {'x': 4}) == {'x': 4}",
        ],
    },
    {
        "id": "top_k_001",
        "prompt": (
            "Write a Python function top_k(xs, k) that returns the k largest values in "
            "descending order."
        ),
        "entrypoint": "top_k",
        "tests": [
            "assert top_k([5, 1, 3, 2], 2) == [5, 3]",
            "assert top_k([1], 5) == [1]",
        ],
    },
    {
        "id": "parse_ints_001",
        "prompt": (
            "Write a Python function parse_ints(text) that returns all signed integers "
            "appearing in a string."
        ),
        "entrypoint": "parse_ints",
        "tests": [
            "assert parse_ints('a -2 and 15 then 0') == [-2, 15, 0]",
            "assert parse_ints('none') == []",
        ],
    },
    {
        "id": "group_by_first_001",
        "prompt": (
            "Write a Python function group_by_first(words) that groups words by their first "
            "character."
        ),
        "entrypoint": "group_by_first",
        "tests": [
            (
                "assert group_by_first(['apple', 'ape', 'bat']) == "
                "{'a': ['apple', 'ape'], 'b': ['bat']}"
            ),
            "assert group_by_first([]) == {}",
        ],
    },
    {
        "id": "transpose_001",
        "prompt": (
            "Write a Python function transpose(matrix) that transposes a rectangular list of "
            "lists."
        ),
        "entrypoint": "transpose",
        "tests": [
            "assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]",
            "assert transpose([]) == []",
        ],
    },
    {
        "id": "balanced_parens_001",
        "prompt": (
            "Write a Python function balanced_parens(text) that returns True if parentheses "
            "are balanced."
        ),
        "entrypoint": "balanced_parens",
        "tests": [
            "assert balanced_parens('(a(b)c)') is True",
            "assert balanced_parens('(()') is False",
        ],
    },
    {
        "id": "running_sum_001",
        "prompt": "Write a Python function running_sum(xs) that returns cumulative sums.",
        "entrypoint": "running_sum",
        "tests": [
            "assert running_sum([1, 2, 3]) == [1, 3, 6]",
            "assert running_sum([]) == []",
        ],
    },
    {
        "id": "roman_small_001",
        "prompt": (
            "Write a Python function roman_to_int(s) that converts roman numerals using "
            "I,V,X,L,C,D,M."
        ),
        "entrypoint": "roman_to_int",
        "tests": [
            "assert roman_to_int('III') == 3",
            "assert roman_to_int('MCMXCIV') == 1994",
        ],
    },
    {
        "id": "slugify_001",
        "prompt": (
            "Write a Python function slugify(text) that lowercases text and joins "
            "alphanumeric word groups with hyphens."
        ),
        "entrypoint": "slugify",
        "tests": [
            "assert slugify('Hello, World!') == 'hello-world'",
            "assert slugify('  A__B  ') == 'a-b'",
        ],
    },
    {
        "id": "apply_patch_paths_001",
        "prompt": (
            "Write a Python function patch_paths(lines) that extracts file paths from lines "
            "beginning with '+++ b/'."
        ),
        "entrypoint": "patch_paths",
        "tests": [
            (
                "assert patch_paths(['+++ b/src/app.py', '--- a/src/app.py', "
                "'+++ /dev/null']) == ['src/app.py']"
            ),
            "assert patch_paths([]) == []",
        ],
    },
]


def main() -> None:
    out = Path("data/prompts/python_smoke.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in PROMPTS:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(PROMPTS)} prompts to {out}")


if __name__ == "__main__":
    main()
