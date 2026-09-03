"""Cross-encoder reranking stage over the fused retrieval candidates.

Hybrid retrieval (dense + BM25) returns a candidate set ranked by a bi-encoder
and lexical overlap; a cross-encoder then rescores each (query, passage) pair
jointly, catching relevance the first stage ranks just below the top. This is
the third IR stage and the second *trained* model (see `frag.train.train_reranker`).

`RerankingStore` wraps any DocumentStore: it over-fetches candidates from the
inner store, reranks, and truncates to `top_k`, so the `search()` contract is
unchanged and actor/critic/harness need no edit. The model is injectable, so the
logic is unit-tested with a fake scorer and no model download. Enabled with
RERANK=on; the trained artifact is selected via RERANKER_MODEL.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Rescore (query, passage) pairs with a cross-encoder."""

    def __init__(self, model: Any | None = None, model_name: str | None = None) -> None:
        self._model = model
        self._model_name = model_name or os.getenv("RERANKER_MODEL", DEFAULT_RERANKER)

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers.cross_encoder import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, query: str, hits: list[dict[str, Any]], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Return `hits` reordered by cross-encoder score (descending).

        Each returned hit's `score` is replaced with the rerank score, and a
        `retrieval_score` key preserves the original for transparency.
        """
        if not hits:
            return []
        scores = self.model.predict([[query, h.get("text", "")] for h in hits])
        ranked = sorted(zip(hits, scores, strict=True), key=lambda hs: hs[1], reverse=True)
        out = []
        for hit, score in ranked:
            h = dict(hit)
            h["retrieval_score"] = h.get("score")
            h["score"] = float(score)
            out.append(h)
        return out[:top_k] if top_k else out


class RerankingStore:
    """DocumentStore decorator: over-fetch from `inner`, then cross-encoder rerank."""

    def __init__(self, inner: Any, reranker: CrossEncoderReranker, candidate_multiplier: int = 5):
        self.inner = inner
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier

    def ingest(self, docs: list[dict[str, Any]]) -> int:
        return self.inner.ingest(docs)

    def count(self) -> int:
        return self.inner.count()

    def search(
        self, query: str, top_k: int = 8, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        candidates = self.inner.search(
            query,
            top_k=max(top_k * self.candidate_multiplier, top_k),
            metadata_filter=metadata_filter,
        )
        return self.reranker.rerank(query, candidates, top_k=top_k)


def rerank_enabled() -> bool:
    return os.getenv("RERANK", "off").strip().lower() in {"on", "1", "true", "yes"}
