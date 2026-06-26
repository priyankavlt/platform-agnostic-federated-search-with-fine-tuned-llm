"""Merge the LoRA adapter into 16-bit weights (for vLLM serving without --enable-lora)."""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/qwen3.8-27b-planner/adapter")
    ap.add_argument("--out", default="outputs/qwen3.8-27b-planner/merged")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    args = ap.parse_args()

    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(args.adapter, max_seq_length=args.max_seq_length, load_in_4bit=True)
    model.save_pretrained_merged(args.out, processor, save_method="merged_16bit")
    print(f"merged model -> {args.out}")


if __name__ == "__main__":
    main()
