"""Explainability: ground an answer in its supporting passages (deterministic).

Reuses the content-relevance infra: an answer's numeric facts are matched against
each retrieved passage, so we can say which passage supports the answer — and flag
answers whose facts appear in no retrieved passage (possible hallucination).
"""

from __future__ import annotations

from typing import Any

from frag.eval.relevance import _normalize, extract_facts


def attribute_rrf(
    doc_key: str, dense_ranking: list[str], sparse_ranking: list[str]
) -> dict[str, Any]:
    """Explain a fused hit: its 1-based rank in each retriever (None if absent)."""
    dense_rank = dense_ranking.index(doc_key) + 1 if doc_key in dense_ranking else None
    sparse_rank = sparse_ranking.index(doc_key) + 1 if doc_key in sparse_ranking else None
    if dense_rank and (sparse_rank is None or dense_rank <= sparse_rank):
        won_on = "dense"
    elif sparse_rank:
        won_on = "bm25"
    else:
        won_on = "none"
    return {"dense_rank": dense_rank, "bm25_rank": sparse_rank, "won_on": won_on}


def ground_answer(
    answer: str, contexts: list[dict[str, Any]], doc_id_fn
) -> dict[str, Any]:
    """Map each fact in the answer to the passages that contain it."""
    facts = extract_facts(answer)
    per_fact: list[dict[str, Any]] = []
    supported = 0
    for fact in facts:
        labels = [
            doc_id_fn(c, i)
            for i, c in enumerate(contexts)
            if fact in _normalize(c.get("text", ""))
        ]
        if labels:
            supported += 1
        per_fact.append({"fact": fact, "supported_by": labels})
    return {
        "facts": per_fact,
        "grounded_facts": supported,
        "total_facts": len(facts),
        # A fact with no supporting passage is a hallucination risk.
        "ungrounded": [f["fact"] for f in per_fact if not f["supported_by"]],
    }
