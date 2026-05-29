import json

from densify.run_artifacts import append_jsonl, new_run_dir, write_json, write_yaml


def test_new_run_dir_creates_unique_prefixed_dir(tmp_path):
    run_dir = new_run_dir(tmp_path, prefix="teacher")
    assert run_dir.exists()
    assert run_dir.name.startswith("teacher_")


def test_write_json_and_append_jsonl(tmp_path):
    json_path = tmp_path / "summary.json"
    jsonl_path = tmp_path / "rows.jsonl"

    write_json(json_path, {"ok": True})
    append_jsonl(jsonl_path, {"id": "a"})
    append_jsonl(jsonl_path, {"id": "b"})

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": "a"}, {"id": "b"}]


def test_write_yaml(tmp_path):
    yaml_path = tmp_path / "config.yaml"

    write_yaml(yaml_path, {"model_id": "fake/model", "nested": {"ok": True}})

    assert "model_id: fake/model" in yaml_path.read_text(encoding="utf-8")
