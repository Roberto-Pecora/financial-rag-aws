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
