"""Explainability: answer grounding and RRF retrieval attribution."""

from __future__ import annotations

from frag.rag import explain


def _doc_id(ctx, idx):
    return ctx.get("metadata", {}).get("doc_id", f"doc-{idx + 1}")


# -- answer grounding ------------------------------------------------------


def test_ground_answer_maps_facts_to_passages():
    contexts = [
        {"text": "Total revenue was 5,678 last year.", "metadata": {"doc_id": "ACME_10K"}},
        {"text": "Unrelated risk factors.", "metadata": {"doc_id": "BETA_10K"}},
    ]
    out = explain.ground_answer("Revenue was 5,678.", contexts, _doc_id)
    assert out["grounded_facts"] == 1 and out["total_facts"] == 1
    assert out["facts"][0]["supported_by"] == ["ACME_10K"]
    assert out["ungrounded"] == []


def test_ground_answer_flags_unsupported_fact():
    contexts = [{"text": "Revenue was 5,678.", "metadata": {"doc_id": "ACME_10K"}}]
    out = explain.ground_answer("Net income was 9,999.", contexts, _doc_id)
    assert out["ungrounded"] == ["9,999"]  # no supporting passage


# -- retrieval attribution -------------------------------------------------


def test_attribute_rrf_won_on_bm25():
    a = explain.attribute_rrf("d1", dense_ranking=["d2", "d1"], sparse_ranking=["d1"])
    assert a == {"dense_rank": 2, "bm25_rank": 1, "won_on": "bm25"}


def test_attribute_rrf_won_on_dense():
    a = explain.attribute_rrf("d1", dense_ranking=["d1"], sparse_ranking=["d2", "d1"])
    assert a["dense_rank"] == 1 and a["won_on"] == "dense"


def test_attribute_rrf_absent_from_sparse():
    a = explain.attribute_rrf("d1", dense_ranking=["d1"], sparse_ranking=["d2"])
    assert a["bm25_rank"] is None and a["won_on"] == "dense"
