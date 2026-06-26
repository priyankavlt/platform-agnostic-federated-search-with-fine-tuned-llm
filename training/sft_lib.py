"""Shared fine-tuning + planner-evaluation logic, used by both ``train_qlora.py`` (CLI on any GPU box) and
``notebooks/finetune_planner_colab.ipynb`` (Colab / Kaggle / Jupyter).

Follows Unsloth's Qwen3.8 recipe: ``FastModel`` + pre-quantised bnb-4bit weights + LoRA on the language
layers only (Qwen3.8 is a vision-language model; vision layers stay frozen for this text-only task).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Qwen chat-template markers used to mask everything except the assistant turn from the loss.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


@dataclass
class TrainConfig:
    base_model: str = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
    max_seq_length: int = 1024
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    train_file: str = "data/sft/train.jsonl"
    val_file: str = "data/sft/val.jsonl"
    output_dir: str = "outputs/qwen3.8-27b-planner"
    epochs: float = 3
    max_steps: int = -1  # >0 overrides epochs (handy for smoke tests)
    per_device_batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.01
    eval_steps: int = 50
    seed: int = 3407
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path, profile: str | None = None) -> TrainConfig:
        raw = yaml.safe_load(Path(path).read_text())
        prof = raw["profiles"][profile or raw["default_profile"]]
        known = {k: v for k, v in prof.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------
def load_model(cfg: TrainConfig):
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        full_finetuning=False,
        dtype=None,
    )
    # Vision-language checkpoints return a processor; text-only SFT needs its inner tokenizer.
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"  # batched generation; training ignores this
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def add_lora(model, cfg: TrainConfig):
    from unsloth import FastModel

    return FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )


# ---------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------
def render_chat(tokenizer, messages: list[dict], generation_prompt: bool = False) -> str:
    # enable_thinking=False: the planner should answer with JSON directly, no <think> block.
    # Templates that don't use the variable simply ignore it.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=False
    )


def load_sft_dataset(tokenizer, cfg: TrainConfig):
    from datasets import load_dataset

    ds = load_dataset("json", data_files={"train": cfg.train_file, "val": cfg.val_file})
    ds = ds.map(lambda b: {"text": [render_chat(tokenizer, m) for m in b["messages"]]},
                batched=True, remove_columns=["messages"])
    sample = ds["train"][0]["text"]
    if INSTRUCTION_PART not in sample or RESPONSE_PART not in sample:
        raise ValueError(
            "Chat template markers not found; update INSTRUCTION_PART/RESPONSE_PART in sft_lib.py.\n"
            f"Rendered sample:\n{sample[:1500]}"
        )
    return ds


# ---------------------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------------------
def make_trainer(model, tokenizer, ds, cfg: TrainConfig):
    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    tokenizer.padding_side = "right"
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        args=SFTConfig(
            output_dir=cfg.output_dir,
            dataset_text_field="text",
            max_length=cfg.max_seq_length,
            num_train_epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.per_device_batch_size,
            per_device_eval_batch_size=cfg.per_device_batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler,
            weight_decay=cfg.weight_decay,
            optim="adamw_8bit",
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_strategy="steps",
            save_steps=cfg.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            seed=cfg.seed,
            report_to="none",
        ),
    )
    return train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)


def save_adapter(model, tokenizer, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


# ---------------------------------------------------------------------------------------------------------
# Planner evaluation (generation + scoring against gold QueryPlans)
# ---------------------------------------------------------------------------------------------------------
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_plan(text: str):
    """Extract a QueryPlan from raw model output; None if it isn't valid JSON matching the schema."""
    from pydantic import ValidationError

    from fedsearch.schemas import QueryPlan

    text = _THINK.sub("", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return QueryPlan.model_validate_json(text[start:end + 1])
    except ValidationError:
        return None


def generate_plans(model, tokenizer, queries: list[str], batch_size: int = 16, max_new_tokens: int = 256) -> list[str]:
    import torch
    from unsloth import FastModel

    from fedsearch.search.prompts import PLANNER_SYSTEM

    FastModel.for_inference(model)
    tokenizer.padding_side = "left"
    outs: list[str] = []
    for i in range(0, len(queries), batch_size):
        prompts = [render_chat(tokenizer, [{"role": "system", "content": PLANNER_SYSTEM},
                                           {"role": "user", "content": q}], generation_prompt=True)
                   for q in queries[i:i + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        outs += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  generated {min(i + batch_size, len(queries))}/{len(queries)}", end="\r")
    FastModel.for_training(model)
    return outs


def score_plans(examples: list[dict], raw_outputs: list[str]) -> dict:
    """Same metrics as ``fedsearch eval``'s planner layer, plus breakdowns by intent and held-out phrasing."""
    from fedsearch.eval.planner_eval import compare_plans, summarize
    from fedsearch.schemas import QueryPlan

    rows, groups = [], defaultdict(list)
    for ex, raw in zip(examples, raw_outputs, strict=True):
        r = compare_plans(parse_plan(raw), QueryPlan.model_validate(ex["plan"]))
        rows.append(r)
        groups[f"intent={ex['intent']}"].append(r)
        groups["held_out_phrasing" if ex.get("test_only") else "seen_phrasing"].append(r)
    return {"overall": summarize(rows), **{k: summarize(v) for k, v in sorted(groups.items())}}


def load_examples(path: str | Path, limit: int | None = None, seed: int = 0) -> list[dict]:
    """Load a query split; with ``limit``, take a fixed random sample (the file is ordered by phrasing type)."""
    import random

    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows
