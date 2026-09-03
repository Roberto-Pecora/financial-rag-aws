import numpy as np


def recall_at_k(pred_ids, gold_ids, k):
    return 1.0 if any(i in pred_ids[:k] for i in gold_ids) else 0.0


def precision_at_k(pred_ids, gold_ids, k):
    return sum(i in gold_ids for i in pred_ids[:k]) / max(k, 1)


def mrr_at_k(pred_ids, gold_ids, k):
    for i, pid in enumerate(pred_ids[:k], 1):
        if pid in gold_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(pred_ids, gold_ids, k):
    dcg = sum(1.0 / np.log2(i + 1) for i, pid in enumerate(pred_ids[:k], 1) if pid in gold_ids)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gold_ids), k) + 1))
    return dcg / idcg if idcg else 0.0
