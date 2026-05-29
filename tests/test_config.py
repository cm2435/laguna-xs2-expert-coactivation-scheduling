from densify.config import load_teacher_smoke_config


def test_load_teacher_smoke_config_defaults_to_decompressed_compressed_tensors(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model_id: fake/model
prompt_path: data/prompts/python_smoke.jsonl
output_dir: runs/teacher_smoke
generation:
  max_new_tokens: 16
""",
        encoding="utf-8",
    )

    cfg = load_teacher_smoke_config(path)

    assert cfg.compressed_tensors_run_compressed is False
    assert cfg.generation.do_sample is True
    assert cfg.generation.temperature == 0.7
    assert cfg.generation.enable_thinking is True


def test_load_teacher_smoke_config_reads_official_sampling_settings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model_id: fake/model
prompt_path: data/prompts/python_smoke.jsonl
output_dir: runs/teacher_smoke
generation:
  max_new_tokens: 16
  temperature: 0.7
  top_k: 20
  top_p: 0.95
  do_sample: true
  enable_thinking: true
""",
        encoding="utf-8",
    )

    cfg = load_teacher_smoke_config(path)

    assert cfg.generation.top_k == 20
    assert cfg.generation.do_sample is True
    assert cfg.generation.enable_thinking is True
