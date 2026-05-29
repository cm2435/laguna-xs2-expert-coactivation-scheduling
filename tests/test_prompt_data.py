import json

from densify.prompt_data import load_jsonl_prompts


def test_load_jsonl_prompts(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "sort_001",
                "prompt": "Write sort_numbers.",
                "entrypoint": "sort_numbers",
                "tests": ["assert sort_numbers([2, 1]) == [1, 2]"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = load_jsonl_prompts(path)

    assert prompts[0].id == "sort_001"
    assert prompts[0].entrypoint == "sort_numbers"
    assert prompts[0].tests == ["assert sort_numbers([2, 1]) == [1, 2]"]
