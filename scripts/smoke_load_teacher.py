from __future__ import annotations

import argparse
from pathlib import Path

from densify.config import load_teacher_smoke_config
from densify.model_introspection import architecture_summary
from densify.run_artifacts import write_json
from densify.teacher_loader import load_teacher_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_teacher_smoke_config(args.config)
    tokenizer = load_tokenizer(cfg.model_id, cfg.trust_remote_code)
    model = load_teacher_model(
        cfg.model_id,
        torch_dtype=cfg.torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
        device_map=cfg.device_map,
    )
    summary = architecture_summary(model, cfg.model_id, cfg.torch_dtype)
    out_dir = Path(cfg.output_dir) / "load_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "architecture.json", summary)

    print(f"model_id={cfg.model_id}")
    print(f"tokenizer_vocab={len(tokenizer)}")
    print(f"model_class={model.__class__.__name__}")
    print(f"num_parameters={summary['num_parameters_seen']}")
    print(f"candidate_moe_modules={len(summary['candidate_moe_modules'])}")
    print(f"wrote={out_dir / 'architecture.json'}")


if __name__ == "__main__":
    main()
