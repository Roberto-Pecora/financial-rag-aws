from __future__ import annotations

import os
import re
import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# RRF constant. 60 is the value from the original Cormack et al. paper and the
# de facto default across search engines; results are insensitive to it.
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenization for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _reciprocal_rank_fusion(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Fuse several ranked lists of doc_ids into a single {doc_id: score} map.

    Each list contributes 1 / (k + rank) to a doc_id's score (rank 0-based),
    so a doc that ranks highly in either retriever floats up. Empty doc_ids
    are ignored.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id:
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class QdrantStore:
    """A document store implementation using Qdrant for vector similarity search.

    Supports two retrieval modes, selected by the RETRIEVAL_MODE env var:
      * "dense" (default): pure vector similarity via Qdrant.
      * "hybrid": dense vectors fused with an in-memory BM25 lexical index
        (built lazily from the collection's chunk text) via Reciprocal Rank
        Fusion, so exact-term matches (e.g. specific figures) can rescue
        chunks that dense embeddings rank just below the top.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        embedding_model: str | None = None,
        retrieval_mode: str | None = None,
    ) -> None:
        self.retrieval_mode = (retrieval_mode or os.getenv("RETRIEVAL_MODE", "dense")).lower()
        # Lazily-built BM25 index state (only populated in hybrid mode).
        self._bm25 = None
        self._bm25_doc_ids: list[str] = []
        self._bm25_payloads: list[dict[str, Any]] = []

        qdrant_url = url or os.getenv("QDRANT_URL")
        qdrant_api_key = api_key or os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY are required")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "sec_filings")

        model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedder = SentenceTransformer(model_name)

        self._ensure_collection()
        self._ensure_payload_indexes()

    def _ensure_collection(self) -> None:
        """Ensure that the Qdrant collection exists, creating it if necessary."""
        try:
            exists = self.client.collection_exists(self.collection_name)
            if isinstance(exists, bool):
                if exists:
                    return
            else:
                if getattr(exists, "exists", False):
                    return
        except Exception:
            pass

        dim = self.embedder.get_sentence_embedding_dimension()
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

    def _ensure_payload_indexes(self) -> None:
        """Ensure that the necessary payload indexes exist for efficient filtering."""
        index_specs = {
            "ticker": models.PayloadSchemaType.KEYWORD,
            "cik": models.PayloadSchemaType.KEYWORD,
            "type": models.PayloadSchemaType.KEYWORD,
            "form_type": models.PayloadSchemaType.KEYWORD,
            "series_id": models.PayloadSchemaType.KEYWORD,
            "source_type": models.PayloadSchemaType.KEYWORD,
            "provider": models.PayloadSchemaType.KEYWORD,
            "filing_date": models.PayloadSchemaType.DATETIME,
        }

        for field_name, schema in index_specs.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                pass

    def ingest(self, docs: list[dict[str, Any]]) -> int:
        """Ingest a list of documents into the Qdrant collection."""
        if not docs:
            return 0

        texts: list[str] = []
        payloads: list[dict[str, Any]] = []
        ids: list[str] = []

        for d in docs:
            text = d.get("text") or d.get("content")
            if not text:
                continue

            doc_id = str(d.get("id") or uuid.uuid4())
            ids.append(doc_id)
            meta = d.get("metadata", {}) or {}
            payload = {"text": text, "doc_id": doc_id, **meta}

            texts.append(text)
            payloads.append(payload)

        if not texts:
            return 0

        vectors = self.embedder.encode(texts, convert_to_numpy=True)

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, ids[i])),
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(texts))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 8,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the collection, dispatching on the configured retrieval mode."""
        if not query:
            return []
        if self.retrieval_mode == "hybrid":
            return self._hybrid_search(query, top_k, metadata_filter)
        return self._dense_search(query, top_k, metadata_filter)

    def _dense_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        query_vec = self.embedder.encode(query, convert_to_numpy=True)

        q_filter = None
        if metadata_filter:
            must = [
                models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in metadata_filter.items()
            ]
            q_filter = models.Filter(must=must)

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )

        results: list[dict[str, Any]] = []
        for h in getattr(hits, "points", []) or []:
            payload = getattr(h, "payload", None) or {}
            results.append(self._format_hit(payload, getattr(h, "score", None)))
        return results

    @staticmethod
    def _format_hit(payload: dict[str, Any], score: Any) -> dict[str, Any]:
        text = payload.get("text", "")
        meta = {k: v for k, v in payload.items() if k != "text"}
        return {"text": text, "metadata": meta, "score": score}

    def _ensure_bm25_index(self) -> None:
        """Build the in-memory BM25 index from the collection's chunk text.

        Scrolls the whole collection once and caches it; only invoked in
        hybrid mode, so pure-dense usage pays none of this cost.
        """
        if self._bm25 is not None:
            return

        from rank_bm25 import BM25Okapi

        payloads: list[dict[str, Any]] = []
        next_page = None
        while True:
            points, next_page = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=next_page,
                with_payload=True,
            )
            for p in points:
                payload = getattr(p, "payload", None) or {}
                if payload.get("text"):
                    payloads.append(payload)
            if next_page is None:
                break

        self._bm25_payloads = payloads
        self._bm25_doc_ids = [p.get("doc_id", "") for p in payloads]
        tokenized = [_tokenize(p.get("text", "")) for p in payloads]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        # Over-fetch from each retriever so fusion has enough candidates to
        # re-order before truncating to top_k.
        candidate_k = max(top_k * 4, 30)

        dense_hits = self._dense_search(query, candidate_k, metadata_filter)
        dense_ranking = [h["metadata"].get("doc_id", "") for h in dense_hits]

        sparse_ranking = self._bm25_ranking(query, candidate_k, metadata_filter)

        scores = _reciprocal_rank_fusion([dense_ranking, sparse_ranking])

        # Payloads for any doc_id we surfaced, from whichever side had it.
        payload_by_id: dict[str, dict[str, Any]] = {}
        for h in dense_hits:
            payload_by_id[h["metadata"].get("doc_id", "")] = {
                "text": h["text"],
                **h["metadata"],
            }
        for doc_id, payload in zip(self._bm25_doc_ids, self._bm25_payloads, strict=True):
            payload_by_id.setdefault(doc_id, payload)

        ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]
        return [
            self._format_hit(payload_by_id.get(doc_id, {}), scores[doc_id]) for doc_id in ranked_ids
        ]

    def _bm25_ranking(
        self,
        query: str,
        candidate_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[str]:
        self._ensure_bm25_index()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        ranking: list[str] = []
        for i in order:
            payload = self._bm25_payloads[i]
            if metadata_filter and not all(payload.get(k) == v for k, v in metadata_filter.items()):
                continue
            ranking.append(self._bm25_doc_ids[i])
            if len(ranking) >= candidate_k:
                break
        return ranking

    def count(self) -> int:
        """Count the number of documents in the Qdrant collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0
