import importlib.util
from pathlib import Path


def load_probe_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_base_tool_probe.py"
    spec = importlib.util.spec_from_file_location("run_base_tool_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_probe_module()


def test_summarize_generation_accepts_tagged_tool_call() -> None:
    text = (
        "<think>Need to inspect.</think>\n"
        "<tool_call>shell\n"
        "<arg_key>command</arg_key>\n"
        "<arg_value>find . -name validators.py</arg_value>\n"
        "</tool_call>"
    )
    summary = probe.summarize_generation(text)
    assert summary["parseable_tool_call"] is True
    assert summary["tool_name"] == "shell"
    assert summary["has_required_arg"] is True


def test_summarize_generation_rejects_no_tool_call() -> None:
    summary = probe.summarize_generation("the input is a list of strings")
    assert summary["parseable_tool_call"] is False
    assert summary["tool_name"] is None
