from frag.eval.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_metrics():
    pred = ["a", "b", "c"]
    gold = ["b"]
    assert recall_at_k(pred, gold, 3) == 1.0
    assert precision_at_k(pred, gold, 3) == 1 / 3
    assert mrr_at_k(pred, gold, 3) == 0.5
    assert ndcg_at_k(pred, gold, 3) > 0
