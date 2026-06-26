import math

from fedsearch.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_ranking_metrics():
    ranked = ["a", "x", "b", "y"]
    rel = {"a", "b", "c"}
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert recall_at_k(ranked, rel, 4) == 2 / 3
    assert reciprocal_rank(["x", "a"], rel) == 0.5
    assert reciprocal_rank(["x"], rel) == 0.0
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(ndcg_at_k(ranked, rel, 4), (1 + 1 / math.log2(4)) / ideal)
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == 1.0
