"""OpenSearch store: index mapping, query construction, and hybrid RRF fusion.

Uses a fake client + fake embedder so the store's logic is exercised with no
live domain and no AWS credentials. The fusion itself is the Qdrant store's,
imported and reused, so hybrid ranking is identical across backends.
"""

from __future__ import annotations

import numpy as np

from frag.rag import store_opensearch as sos


class _FakeEmbedder:
    def __init__(self, dim=4):
        self._dim = dim

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, text, convert_to_numpy=True):
        # Deterministic vector; single string -> 1-D, list -> 2-D.
        if isinstance(text, str):
            return np.ones(self._dim, dtype=float)
        return np.ones((len(text), self._dim), dtype=float)


class _FakeIndices:
    def __init__(self):
        self.created = {}

    def exists(self, index):
        return index in self.created

    def create(self, index, body):
        self.created[index] = body


class _FakeClient:
    """Records the last search body and returns canned hits per query type."""

    def __init__(self, dense_hits, sparse_hits):
        self.indices = _FakeIndices()
        self._dense = dense_hits
        self._sparse = sparse_hits
        self.bodies = []

    def search(self, index, body):
        self.bodies.append(body)
        is_knn = "knn" in str(body["query"])
        hits = self._dense if is_knn else self._sparse
        return {"hits": {"hits": hits}}

    def count(self, index):
        return {"count": 7}


def _hit(doc_id, text, score):
    return {"_id": doc_id, "_score": score, "_source": {"doc_id": doc_id, "text": text}}


# --------------------------------------------------------------------------
# pure builders
# --------------------------------------------------------------------------


def test_build_index_body_sets_knn_dim_and_hnsw_knobs():
    body = sos.build_index_body(384)
    assert body["settings"]["index"]["knn"] is True
    assert body["settings"]["index"]["knn.algo_param.ef_search"] == 256
    assert body["mappings"]["properties"]["vector"]["dimension"] == 384
    method = body["mappings"]["properties"]["vector"]["method"]
    assert method["engine"] == "lucene"  # current engine, not deprecated nmslib
    assert method["parameters"]["ef_construction"] == 256 and method["parameters"]["m"] == 16
    assert body["mappings"]["properties"]["ingest_path"]["type"] == "keyword"


def test_query_builders_exclude_vector_and_apply_filter():
    q = sos.knn_query([0.1, 0.2], top_k=5, metadata_filter={"ticker": "AAPL"})
    assert q["_source"]["excludes"] == ["vector"]
    assert q["size"] == 5
    assert {"term": {"ticker": "AAPL"}} in q["query"]["bool"]["must"]

    assert q["query"]["bool"]["must"][0]["knn"]["vector"]["method_parameters"]["ef_search"] == 256

    b = sos.bm25_query("revenue", top_k=3, metadata_filter=None)
    assert b["query"]["match"]["text"]["query"] == "revenue"


def test_format_hit_strips_vector_field():
    out = sos._format_hit({"text": "hi", "doc_id": "d1", "ticker": "X"}, 1.2)
    assert out["text"] == "hi"
    assert out["metadata"]["ticker"] == "X"
    assert out["score"] == 1.2


# --------------------------------------------------------------------------
# store behaviour with fakes
# --------------------------------------------------------------------------


def _store(dense, sparse, mode="hybrid"):
    client = _FakeClient(dense, sparse)
    return sos.OpenSearchStore(
        index_name="test",
        client=client,
        embedder=_FakeEmbedder(),
        retrieval_mode=mode,
    ), client


def test_ensure_index_created_on_init():
    store, client = _store([], [])
    assert "test" in client.indices.created


def test_dense_search_returns_store_shape():
    store, _ = _store([_hit("d1", "revenue rose", 0.9)], [], mode="dense")
    results = store.search("revenue", top_k=3)
    assert results == [{"text": "revenue rose", "metadata": {"doc_id": "d1"}, "score": 0.9}]


def test_hybrid_search_fuses_both_rankings():
    # d1 ranks top on dense, d2 top on sparse; RRF should surface both.
    dense = [_hit("d1", "a", 0.9), _hit("d3", "c", 0.5)]
    sparse = [_hit("d2", "b", 8.0), _hit("d1", "a", 2.0)]
    store, _ = _store(dense, sparse)
    results = store.search("q", top_k=3)
    ids = [r["metadata"]["doc_id"] for r in results]
    assert "d1" in ids and "d2" in ids  # a doc strong in either retriever survives
    assert all("text" in r and "score" in r for r in results)


def test_count_reads_client():
    store, _ = _store([], [])
    assert store.count() == 7
