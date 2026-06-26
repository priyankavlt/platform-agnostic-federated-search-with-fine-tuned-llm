"""Render the labelled query splits into chat-format SFT data for the planner.

    uv run python training/build_sft.py            # -> data/sft/{train,val,test}.jsonl

Each row: {"messages": [system, user, assistant]} where the assistant turn is the gold QueryPlan JSON.
The system prompt is imported from the serving code so train/serve prompts can never drift apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fedsearch.schemas import QueryPlan
from fedsearch.search.prompts import PLANNER_SYSTEM


def to_chat(ex: dict) -> dict:
    plan = QueryPlan.model_validate(ex["plan"])
    return {"messages": [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": ex["query"]},
        {"role": "assistant", "content": plan.model_dump_json()},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, default=Path("data/queries"))
    ap.add_argument("--out", type=Path, default=Path("data/sft"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in (args.queries / f"{split}.jsonl").read_text().splitlines() if line]
        with (args.out / f"{split}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(to_chat(r)) + "\n")
        print(f"{split}: {len(rows)} -> {args.out / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()
