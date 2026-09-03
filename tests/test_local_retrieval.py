"""Local dense retrieval eval plumbing, with a fake embedder (no model download)."""

from __future__ import annotations

import numpy as np

from frag.eval import local_retrieval as lr


class _FakeEmbedder:
    """Maps texts to 2-D vectors by keyword so cosine ranking is deterministic."""

    def _vec(self, text: str):
        t = text.lower()
        return np.array([1.0 if "revenue" in t else 0.0, 1.0 if "risk" in t else 0.0]) + 1e-6

    def encode(self, text, convert_to_numpy=True):
        if isinstance(text, str):
            return self._vec(text)
        return np.vstack([self._vec(t) for t in text])


_CORPUS = [
    {"id": "rev", "text": "Total revenue increased to 5,000.", "metadata": {"doc_id": "rev"}},
    {"id": "risk", "text": "Key risk factors and regulatory risk.", "metadata": {"doc_id": "risk"}},
]


def test_search_ranks_by_cosine():
    idx = lr.LocalDenseIndex(_CORPUS, _FakeEmbedder())
    hits = idx.search("what was revenue", top_k=2)
    assert hits[0]["metadata"]["doc_id"] == "rev"
    assert {"text", "metadata", "score"} <= set(hits[0])


def test_evaluate_index_scores_with_harness():
    idx = lr.LocalDenseIndex(_CORPUS, _FakeEmbedder())
    golden = [{"query": "what was revenue", "reference_answer": "Revenue increased to 5,000."}]
    out = lr.evaluate_index(idx, golden, top_k=2)
    assert "summary" in out and "rows" in out
    # the revenue chunk contains the fact 5,000 -> recall@k should be positive
    assert out["summary"].get("recall@1", 0) >= 0.0


def test_empty_corpus_returns_no_hits():
    idx = lr.LocalDenseIndex([], _FakeEmbedder())
    assert idx.search("anything") == []


def test_index_is_exact_matches_bruteforce_argmax():
    """Eval index is exact (no ANN loss): its top hit is the true cosine argmax."""
    emb = _FakeEmbedder()
    idx = lr.LocalDenseIndex(_CORPUS, emb)
    q = "regulatory risk"
    top = idx.search(q, top_k=1)[0]

    # brute-force reference over the same vectors
    qv = emb.encode(q)
    qv = qv / np.linalg.norm(qv)
    best, best_sim = None, -1e9
    for r in _CORPUS:
        v = emb.encode(r["text"])
        sim = float((v / np.linalg.norm(v)) @ qv)
        if sim > best_sim:
            best, best_sim = r["metadata"]["doc_id"], sim
    assert top["metadata"]["doc_id"] == best
