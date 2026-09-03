"""Index-axis ablation (exact vs quantised) with fake indexes; and turbovec plumbing."""

from __future__ import annotations

import numpy as np

from frag.eval import ablation
from frag.eval.local_retrieval import LocalDenseIndex
from frag.rag.store_turbovec import RescoringIndex, TurbovecIndex


class _FakeEmbedder:
    def _vec(self, t):
        t = t.lower()
        return np.array([1.0 if "revenue" in t else 0.0, 1.0 if "risk" in t else 0.0]) + 1e-6

    def encode(self, text, convert_to_numpy=True):
        return self._vec(text) if isinstance(text, str) else np.vstack([self._vec(x) for x in text])

    def get_sentence_embedding_dimension(self):
        return 2


class _FakeTurbovecIndex:
    """Mimics turbovec's add/search over stored vectors (exact here, for testing)."""

    def __init__(self):
        self._vecs = None

    def add(self, vectors):
        self._vecs = np.asarray(vectors, dtype="float32")

    def search(self, q, k):
        sims = self._vecs @ np.asarray(q, dtype="float32")
        order = np.argsort(-sims)[:k]
        return sims[order], order


_CORPUS = [
    {"id": "rev", "text": "Total revenue increased to 5,000.", "metadata": {"doc_id": "rev"}},
    {"id": "risk", "text": "Key risk factors remain.", "metadata": {"doc_id": "risk"}},
]
_GOLDEN = [{"query": "what was revenue", "reference_answer": "Revenue increased to 5,000."}]


def test_turbovec_index_search_contract():
    idx = TurbovecIndex(_CORPUS, _FakeEmbedder(), bit_width=4, index=_FakeTurbovecIndex())
    hits = idx.search("what was revenue", top_k=1)
    assert hits[0]["metadata"]["doc_id"] == "rev"
    assert idx.memory_bytes == len(_CORPUS) * 2 * 4 // 8


def test_rescoring_index_reorders_by_full_precision():
    first = TurbovecIndex(_CORPUS, _FakeEmbedder(), index=_FakeTurbovecIndex())
    rescored = RescoringIndex(first, _FakeEmbedder(), candidate_multiplier=5)
    hits = rescored.search("what was revenue", top_k=1)
    assert hits[0]["metadata"]["doc_id"] == "rev"


def test_run_index_ablation_reports_latency_and_memory():
    exact = LocalDenseIndex(_CORPUS, _FakeEmbedder())
    turbo = TurbovecIndex(_CORPUS, _FakeEmbedder(), index=_FakeTurbovecIndex())
    logged = []
    results = ablation.run_index_ablation(
        {"exact float32": exact, "turbovec-4bit": turbo},
        _GOLDEN,
        top_k=2,
        repeats=2,
        mlflow_logger=lambda c: logged.append(c["name"]),
    )
    assert [r["name"] for r in results] == ["exact float32", "turbovec-4bit"]
    assert all("p50_ms" in r["latency"] for r in results)
    assert results[1]["memory_bytes"] > 0
    assert logged == ["exact float32", "turbovec-4bit"]
    md = ablation.format_index_markdown(results)
    assert "mem MB" in md and "| Index |" in md
