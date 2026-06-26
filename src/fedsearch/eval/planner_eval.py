"""Structured-output accuracy of the query planner against gold QueryPlans."""

from __future__ import annotations

from fedsearch.ingest.entities import norm
from fedsearch.schemas import QueryPlan


def _ents(p: QueryPlan) -> set[tuple[str, str]]:
    return {(e.type, norm(e.value)) for e in p.entities}


def compare_plans(pred: QueryPlan | None, gold: QueryPlan) -> dict[str, float]:
    if pred is None:  # schema failure / server error counts as fully wrong
        return {"valid": 0.0, "intent_acc": 0.0, "sources_exact": 0.0, "entities_exact": 0.0, "plan_exact": 0.0,
                "src_tp": 0, "src_fp": 0, "src_fn": len(gold.sources), "ent_tp": 0, "ent_fp": 0,
                "ent_fn": len(gold.entities)}
    ps, gs = set(pred.sources), set(gold.sources)
    pe, ge = _ents(pred), _ents(gold)
    intent = float(pred.intent == gold.intent)
    return {
        "valid": 1.0,
        "intent_acc": intent,
        "sources_exact": float(ps == gs),
        "entities_exact": float(pe == ge),
        "plan_exact": float(intent and ps == gs and pe == ge),
        "src_tp": len(ps & gs), "src_fp": len(ps - gs), "src_fn": len(gs - ps),
        "ent_tp": len(pe & ge), "ent_fp": len(pe - ge), "ent_fn": len(ge - pe),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    n = len(rows) or 1
    tot = {k: sum(r[k] for r in rows) for k in rows[0]} if rows else {}

    def prf(prefix: str) -> dict[str, float]:
        tp, fp, fn = tot.get(f"{prefix}_tp", 0), tot.get(f"{prefix}_fp", 0), tot.get(f"{prefix}_fn", 0)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {f"{prefix}_precision": round(p, 4), f"{prefix}_recall": round(r, 4), f"{prefix}_f1": round(f, 4)}

    return {
        "n": len(rows),
        **{k: round(tot.get(k, 0) / n, 4) for k in ("valid", "intent_acc", "sources_exact", "entities_exact", "plan_exact")},
        **prf("src"),
        **prf("ent"),
    }
