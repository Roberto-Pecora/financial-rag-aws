"""Cross-encoder reranking stage: reorder, decorator over-fetch, env toggle."""

from __future__ import annotations

from frag.rag import reranker as rr


class _FakeModel:
    """predict returns a score per [query, text] pair based on a keyword."""

    def predict(self, pairs):
        return [2.0 if "revenue" in text.lower() else 0.1 for _q, text in pairs]


def _hits():
    return [
        {"text": "Risks and regulatory pressure.", "metadata": {"doc_id": "risk"}, "score": 0.9},
        {"text": "Total revenue rose to 5,000.", "metadata": {"doc_id": "rev"}, "score": 0.5},
    ]


def test_rerank_reorders_by_cross_encoder_score():
    reranker = rr.CrossEncoderReranker(model=_FakeModel())
    out = reranker.rerank("what was revenue", _hits())
    assert out[0]["metadata"]["doc_id"] == "rev"  # promoted despite lower retrieval score
    assert out[0]["score"] == 2.0
    assert out[0]["retrieval_score"] == 0.5  # original preserved


def test_rerank_respects_top_k_and_empty():
    reranker = rr.CrossEncoderReranker(model=_FakeModel())
    assert reranker.rerank("q", _hits(), top_k=1)[0]["metadata"]["doc_id"] == "rev"
    assert reranker.rerank("q", []) == []


class _FakeInner:
    def __init__(self, hits):
        self._hits = hits
        self.last_top_k = None

    def search(self, query, top_k=8, metadata_filter=None):
        self.last_top_k = top_k
        return self._hits

    def ingest(self, docs):
        return len(docs)

    def count(self):
        return 42


def test_reranking_store_overfetches_then_truncates():
    inner = _FakeInner(_hits())
    store = rr.RerankingStore(
        inner, rr.CrossEncoderReranker(model=_FakeModel()), candidate_multiplier=5
    )
    results = store.search("what was revenue", top_k=1)

    assert inner.last_top_k == 5  # over-fetched top_k * multiplier
    assert len(results) == 1
    assert results[0]["metadata"]["doc_id"] == "rev"
    assert store.count() == 42  # delegates


def test_rerank_enabled_env(monkeypatch):
    monkeypatch.setenv("RERANK", "on")
    assert rr.rerank_enabled() is True
    monkeypatch.setenv("RERANK", "off")
    assert rr.rerank_enabled() is False
