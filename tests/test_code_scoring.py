from densify.code_scoring import extract_python_code, score_code_generation


def test_extract_python_code_from_markdown_block():
    text = "Here is code:\n```python\ndef f():\n    return 1\n```"
    assert extract_python_code(text) == "def f():\n    return 1"


def test_score_code_generation_passes_tests():
    score = score_code_generation(
        "```python\ndef add_one(x):\n    return x + 1\n```",
        ["assert add_one(2) == 3"],
    )
    assert score.parse_ok is True
    assert score.tests_ok is True


def test_score_code_generation_reports_failed_tests():
    score = score_code_generation(
        "```python\ndef add_one(x):\n    return x\n```",
        ["assert add_one(2) == 3"],
    )
    assert score.parse_ok is True
    assert score.tests_ok is False
    assert "AssertionError" in score.test_stderr
