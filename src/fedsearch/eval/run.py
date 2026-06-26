"""Evaluation harness over the held-out query split.

Reports three layers separately so regressions are attributable:
  planner   - structured-output accuracy/precision of the (fine-tuned) planner vs gold plans
  retrieval - Recall@K / Precision@K / MRR / nDCG, end-to-end (predicted plan) and oracle (gold plan)
  answers   - groundedness and citation checks
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path

from fedsearch.eval.answer_eval import evaluate_answer
from fedsearch.eval.metrics import mean_dicts, retrieval_metrics
from fedsearch.eval.planner_eval import compare_plans, summarize
from fedsearch.schemas import QueryPlan
from fedsearch.search.engine import SearchEngine


def load_split(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


async def evaluate(engine: SearchEngine, examples: list[dict], k: int = 10, answers: bool = True,
                   concurrency: int = 8) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def one(ex: dict) -> dict:
        gold = QueryPlan.model_validate(ex["plan"])
        relevant = set(ex["relevant_doc_ids"])
        async with sem:
            try:
                pred = await engine.planner.plan(ex["query"])
            except Exception as e:  # noqa: BLE001 - invalid structured output after retries, server error, ...
                pred, err = None, repr(e)
            else:
                err = None
            # On planner failure, fall back to searching everything with the raw query (never the gold plan).
            fallback = QueryPlan(intent="general", sources=[], rewritten_query=ex["query"])
            hits = engine.searcher.search(pred or fallback, k)
            oracle_hits = engine.searcher.search(gold, k)
            ans_m = None
            if answers:
                ans = await engine.answerer.answer(ex["query"], hits)
                ans_m = evaluate_answer(ans, hits, relevant, ex.get("answer_facts", []))
        return {
            "intent": ex["intent"],
            "query": ex["query"],
            "error": err,
            "plan": compare_plans(pred, gold),
            "retrieval": retrieval_metrics([h.doc.id for h in hits], relevant),
            "retrieval_oracle": retrieval_metrics([h.doc.id for h in oracle_hits], relevant),
            "answer": ans_m,
        }

    rows = await asyncio.gather(*(one(ex) for ex in examples))
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_intent[r["intent"]].append(r)

    def block(rs: list[dict]) -> dict:
        out = {
            "planner": summarize([r["plan"] for r in rs]),
            "retrieval": mean_dicts([r["retrieval"] for r in rs]),
            "retrieval_oracle_plan": mean_dicts([r["retrieval_oracle"] for r in rs]),
        }
        if answers:
            out["answers"] = mean_dicts([r["answer"] for r in rs])
        return out

    return {
        "overall": block(rows),
        "by_intent": {i: block(rs) for i, rs in sorted(by_intent.items())},
        "errors": [{"query": r["query"], "error": r["error"]} for r in rows if r["error"]][:20],
        "plan_failures": [r["query"] for r in rows if not r["plan"]["plan_exact"]][:50],
    }
