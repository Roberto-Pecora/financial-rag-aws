"""OpenSearch document store: dense k-NN and BM25 in one managed index.

Replaces the template's QdrantStore with a single AWS OpenSearch domain that
holds both a `knn_vector` field (dense embeddings computed app-side) and the
analysed `text` field (BM25). Hybrid retrieval runs the two queries separately
and fuses their rankings with the *same* Reciprocal Rank Fusion used by the
Qdrant store, so the retrieval behaviour is unchanged and only the backend
moved. The `search()` contract — `{text, metadata, score}` — is identical, so
actor, critic, controller and the eval harness need no change.

The OpenSearch client and the embedder are injectable, so index-body and query
construction are unit-tested with fakes and no live domain.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

# Reuse the exact fusion from the Qdrant store so hybrid ranking is provably
# identical across backends (OpenSearch does its own BM25 analysis server-side,
# so the in-memory tokeniser is not needed here).
from frag.rag.store_qdrant import _reciprocal_rank_fusion

_KEYWORD_FIELDS = (
    "doc_id",
    "ticker",
    "cik",
    "type",
    "form_type",
    "series_id",
    "source",
    "source_type",
    "provider",
    "ingest_path",
)


def build_index_body(dim: int) -> dict[str, Any]:
    """Index mapping: a knn_vector for dense search plus text and keyword fields."""
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                    },
                },
                **{f: {"type": "keyword"} for f in _KEYWORD_FIELDS},
                "filing_date": {"type": "date", "ignore_malformed": True},
                "chunk_index": {"type": "integer"},
            }
        },
    }


def knn_query(vector: list[float], top_k: int, metadata_filter: dict[str, Any] | None) -> dict:
    q: dict[str, Any] = {"knn": {"vector": {"vector": vector, "k": top_k}}}
    return _wrap_filter(q, metadata_filter, top_k)


def bm25_query(query: str, top_k: int, metadata_filter: dict[str, Any] | None) -> dict:
    q: dict[str, Any] = {"match": {"text": {"query": query}}}
    return _wrap_filter(q, metadata_filter, top_k)


def _wrap_filter(inner: dict, metadata_filter: dict[str, Any] | None, top_k: int) -> dict:
    if metadata_filter:
        must = [inner] + [{"term": {k: v}} for k, v in metadata_filter.items()]
        body_query = {"bool": {"must": must}}
    else:
        body_query = inner
    return {"size": top_k, "query": body_query, "_source": {"excludes": ["vector"]}}


def _format_hit(source: dict[str, Any], score: Any) -> dict[str, Any]:
    text = source.get("text", "")
    meta = {k: v for k, v in source.items() if k != "text"}
    return {"text": text, "metadata": meta, "score": score}


class OpenSearchStore:
    """Hybrid (dense + BM25) store over a single OpenSearch index."""

    def __init__(
        self,
        index_name: str | None = None,
        endpoint: str | None = None,
        embedding_model: str | None = None,
        retrieval_mode: str | None = None,
        client: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.retrieval_mode = (retrieval_mode or os.getenv("RETRIEVAL_MODE", "hybrid")).lower()
        self.index_name = index_name or os.getenv("OPENSEARCH_INDEX", "sec_filings")
        self.client = client or _default_client(endpoint)
        self.embedder = embedder or _default_embedder(embedding_model)
        self._ensure_index()

    # -- indexing ----------------------------------------------------------
    def _dim(self) -> int:
        return self.embedder.get_sentence_embedding_dimension()

    def _ensure_index(self) -> None:
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=build_index_body(self._dim()))

    def ingest(self, docs: list[dict[str, Any]]) -> int:
        from opensearchpy.helpers import bulk

        texts, actions = [], []
        for d in docs:
            text = d.get("text") or d.get("content")
            if not text:
                continue
            texts.append(text)
            actions.append(d)
        if not texts:
            return 0

        vectors = self.embedder.encode(texts, convert_to_numpy=True)
        body = []
        for d, vec in zip(actions, vectors, strict=True):
            doc_id = str(d.get("id") or uuid.uuid4())
            meta = d.get("metadata", {}) or {}
            body.append(
                {
                    "_index": self.index_name,
                    "_id": doc_id,  # content-hash id -> idempotent upsert
                    "_source": {
                        "text": d["text"],
                        "doc_id": doc_id,
                        "vector": vec.tolist(),
                        **meta,
                    },
                }
            )
        bulk(self.client, body)
        return len(body)

    # -- retrieval ---------------------------------------------------------
    def search(
        self, query: str, top_k: int = 8, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not query:
            return []
        if self.retrieval_mode == "hybrid":
            return self._hybrid_search(query, top_k, metadata_filter)
        return self._dense_search(query, top_k, metadata_filter)

    def _run(self, body: dict) -> list[dict[str, Any]]:
        resp = self.client.search(index=self.index_name, body=body)
        return resp.get("hits", {}).get("hits", [])

    def _dense_search(self, query, top_k, metadata_filter):
        vec = self.embedder.encode(query, convert_to_numpy=True).tolist()
        hits = self._run(knn_query(vec, top_k, metadata_filter))
        return [_format_hit(h["_source"], h.get("_score")) for h in hits]

    def _hybrid_search(self, query, top_k, metadata_filter):
        candidate_k = max(top_k * 4, 30)
        vec = self.embedder.encode(query, convert_to_numpy=True).tolist()

        dense_hits = self._run(knn_query(vec, candidate_k, metadata_filter))
        sparse_hits = self._run(bm25_query(query, candidate_k, metadata_filter))

        source_by_id: dict[str, dict[str, Any]] = {}
        dense_ranking, sparse_ranking = [], []
        for h in dense_hits:
            did = h["_source"].get("doc_id", h["_id"])
            source_by_id[did] = h["_source"]
            dense_ranking.append(did)
        for h in sparse_hits:
            did = h["_source"].get("doc_id", h["_id"])
            source_by_id.setdefault(did, h["_source"])
            sparse_ranking.append(did)

        scores = _reciprocal_rank_fusion([dense_ranking, sparse_ranking])
        ranked = sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]
        return [_format_hit(source_by_id.get(d, {}), scores[d]) for d in ranked]

    def count(self) -> int:
        return int(self.client.count(index=self.index_name).get("count", 0))


# --------------------------------------------------------------------------
# Default AWS-signed client + embedder (constructed lazily, kept out of tests)
# --------------------------------------------------------------------------


def _default_client(endpoint: str | None):
    import boto3
    from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

    endpoint = endpoint or os.getenv("OPENSEARCH_ENDPOINT")
    if not endpoint:
        raise RuntimeError("OPENSEARCH_ENDPOINT is required")
    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")

    region = os.getenv("AWS_REGION", "eu-north-1")
    session = boto3.Session(profile_name=os.getenv("AWS_PROFILE"), region_name=region)
    auth = AWSV4SignerAuth(session.get_credentials(), region, "es")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )


def _default_embedder(model_name: str | None):
    from sentence_transformers import SentenceTransformer

    name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    return SentenceTransformer(name)
