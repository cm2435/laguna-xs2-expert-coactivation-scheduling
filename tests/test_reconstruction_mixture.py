import importlib.util
import json
from pathlib import Path


def load_mixture_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_reconstruction_mixture.py"
    spec = importlib.util.spec_from_file_location("build_reconstruction_mixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mixture = load_mixture_module()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_build_mixture_interleaves_sources_and_records_source(tmp_path) -> None:
    code = tmp_path / "code.jsonl"
    tools = tmp_path / "tools.jsonl"
    recovery = tmp_path / "recovery.jsonl"
    out = tmp_path / "mixed.jsonl"

    write_jsonl(code, [{"text": "code A"}, {"text": "code B"}])
    write_jsonl(tools, [{"messages": [{"role": "user", "content": "u"}]}])
    write_jsonl(
        recovery,
        [{"messages": [{"role": "assistant", "content": "<tool_call>shell</tool_call>"}]}],
    )

    summary = mixture.build_mixture(
        code_path=code,
        tool_path=tools,
        recovery_path=recovery,
        output_path=out,
        max_rows=4,
        seed=123,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert {row["_reconstruction_source"] for row in rows} == {"code", "tool", "recovery"}
    assert summary["rows_written"] == 4
