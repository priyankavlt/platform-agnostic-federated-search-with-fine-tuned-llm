"""Fine-tune the query planner from the command line on any CUDA machine (same code the Colab notebook uses).

    uv sync --extra train   # on a CUDA machine
    uv run fedsearch seed && uv run python training/build_sft.py
    uv run python training/train_qlora.py --profile qwen3.8-27b-qlora --eval-limit 200

Prints planner accuracy before and after fine-tuning on the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Unsloth must be imported before transformers/trl so its patches apply.
import unsloth  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.sft_lib import (
    TrainConfig,
    add_lora,
    generate_plans,
    load_examples,
    load_model,
    load_sft_dataset,
    make_trainer,
    save_adapter,
    score_plans,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/train_qlora.yaml"))
    ap.add_argument("--profile")
    ap.add_argument("--test-file", default="data/queries/test.jsonl")
    ap.add_argument("--eval-limit", type=int, default=200, help="test examples to score (0 = skip eval)")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config, args.profile)
    model, tokenizer = load_model(cfg)
    test = load_examples(args.test_file, args.eval_limit) if args.eval_limit else []
    report: dict = {"config": cfg.__dict__}

    if test:
        report["before"] = score_plans(test, generate_plans(model, tokenizer, [e["query"] for e in test]))
        print("\nbase model:", json.dumps(report["before"]["overall"], indent=1))

    model = add_lora(model, cfg)
    make_trainer(model, tokenizer, load_sft_dataset(tokenizer, cfg), cfg).train()
    adapter = save_adapter(model, tokenizer, Path(cfg.output_dir) / "adapter")
    print(f"LoRA adapter saved to {adapter}")

    if test:
        report["after"] = score_plans(test, generate_plans(model, tokenizer, [e["query"] for e in test]))
        print("\nfine-tuned:", json.dumps(report["after"]["overall"], indent=1))
    (Path(cfg.output_dir) / "planner_eval.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
