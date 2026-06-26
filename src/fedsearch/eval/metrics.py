"""Ranking metrics with binary relevance."""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / k if k else 0.0


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(ranked: Sequence[str], relevant: set[str], ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    out = {"mrr": reciprocal_rank(ranked, relevant)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        out[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
    return out


def mean_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}
