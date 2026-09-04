"""In-memory dense retrieval eval, so finetune lift is measurable without AWS.

Builds a cosine index over the corpus from any SentenceTransformer - the base
`bge-small` or the finetuned artifact - runs the golden queries through it, and
scores with the existing content-based harness (`frag.eval.harness.evaluate`).
This gives a base-vs-finetuned recall/nDCG comparison locally, before the same
model is ever pushed to the OpenSearch domain.

The embedder is injectable, so the search/prediction plumbing is unit-tested
with a fake model and no downloads.
"""

from __future__ import annotations

from typing import Any

from frag.eval.harness import evaluate
from frag.rag.explain import attribute_rrf
from frag.rag.store_qdrant import _reciprocal_rank_fusion


class LocalDenseIndex:
    """Cosine-similarity search over an in-memory embedded corpus."""

    def __init__(self, corpus: list[dict[str, Any]], embedder: Any) -> None:
        import numpy as np

        self._records = [r for r in corpus if r.get("text")]
        self.embedder = embedder
        if not self._records:
            self._matrix = np.zeros((0, 0), dtype="float32")
            return
        mat = embedder.encode([r["text"] for r in self._records], convert_to_numpy=True)
        mat = np.asarray(mat, dtype="float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._matrix = mat / np.clip(norms, 1e-12, None)

    def search(
        self, query: str, top_k: int = 10, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        import numpy as np

        if not self._records:
            return []
        q = np.asarray(self.embedder.encode(query, convert_to_numpy=True), dtype="float32")
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self._matrix @ q
        order = np.argsort(-sims)[:top_k]
        out = []
        for i in order:
            r = self._records[int(i)]
            out.append(
                {"text": r["text"], "metadata": r.get("metadata", {}), "score": float(sims[int(i)])}
            )
        return out


def predictions_for_golden(
    index: LocalDenseIndex, golden: list[dict[str, Any]], top_k: int = 10
) -> list[dict[str, Any]]:
    """Run each golden query through the index into harness-shaped predictions."""
    preds = []
    for g in golden:
        hits = index.search(g["query"], top_k=top_k)
        preds.append({"query": g["query"], "retrieved_docs": hits, "critic_score": 0.0})
    return preds


def evaluate_index(
    index: LocalDenseIndex, golden: list[dict[str, Any]], top_k: int = 10
) -> dict[str, Any]:
    """Retrieve for every golden query and score with the content-based harness."""
    preds = predictions_for_golden(index, golden, top_k=top_k)
    return evaluate(preds, golden)


class LocalHybridIndex:
    """Dense (exact cosine) + BM25 fused with Reciprocal Rank Fusion, in memory.

    Reuses the exact RRF from the Qdrant store so local hybrid ranking matches the
    production backend. Implements the same search() contract.
    """

    def __init__(self, corpus: list[dict[str, Any]], embedder: Any) -> None:
        import re

        from rank_bm25 import BM25Okapi

        self._records = [r for r in corpus if r.get("text")]
        self.dense = LocalDenseIndex(self._records, embedder)
        self._tokenise = lambda t: re.findall(r"[a-z0-9]+", t.lower())
        self._bm25 = (
            BM25Okapi([self._tokenise(r["text"]) for r in self._records]) if self._records else None
        )
        self._text_to_idx = {r["text"]: i for i, r in enumerate(self._records)}

    def search(
        self, query: str, top_k: int = 10, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        import numpy as np

        if not self._records or self._bm25 is None:
            return []
        candidate_k = max(top_k * 4, 30)

        # Dense ranking (as record indices).
        dense_hits = self.dense.search(query, top_k=candidate_k)
        dense_rank = [str(self._text_to_idx[h["text"]]) for h in dense_hits]

        # BM25 ranking (as record indices).
        scores = self._bm25.get_scores(self._tokenise(query))
        bm25_rank = [str(i) for i in np.argsort(-scores)[:candidate_k]]

        fused = _reciprocal_rank_fusion([dense_rank, bm25_rank])
        ranked = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]
        out = []
        for i in ranked:
            r = self._records[int(i)]
            attribution = attribute_rrf(i, dense_rank, bm25_rank)
            meta = {**r.get("metadata", {}), "_attribution": attribution}
            out.append({"text": r["text"], "metadata": meta, "score": fused[i]})
        return out
