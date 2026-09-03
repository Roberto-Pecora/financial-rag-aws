"""Quantised dense search (turbovec) as a measured efficiency option, not the default.

turbovec applies 2/4-bit TurboQuant compression (~16x smaller), trading some
recall for memory/latency. It never ranks alone here: it is a first-stage recall
that a full-precision rescore (RescoringIndex) reorders, recovering most of the
quantisation loss. The turbovec index is injectable so this is tested with a fake.
"""

from __future__ import annotations

import os
from typing import Any


class TurbovecIndex:
    """Dense search over a turbovec quantised index; implements search()."""

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        embedder: Any,
        bit_width: int = 4,
        index: Any | None = None,
    ) -> None:
        self._records = [r for r in corpus if r.get("text")]
        self.embedder = embedder
        self.bit_width = bit_width
        dim = embedder.get_sentence_embedding_dimension()
        self.index = index or self._build_index(dim, bit_width)
        if self._records:
            vectors = embedder.encode([r["text"] for r in self._records], convert_to_numpy=True)
            self.index.add(vectors)

    @staticmethod
    def _build_index(dim: int, bit_width: int):
        from turbovec import TurboQuantIndex

        return TurboQuantIndex(dim=dim, bit_width=bit_width)

    def search(
        self, query: str, top_k: int = 8, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not self._records:
            return []
        q = self.embedder.encode(query, convert_to_numpy=True)
        _scores, indices = self.index.search(q, k=top_k)
        out = []
        for rank, i in enumerate(indices):
            r = self._records[int(i)]
            out.append(
                {"text": r["text"], "metadata": r.get("metadata", {}), "score": 1.0 / (rank + 1)}
            )
        return out

    @property
    def memory_bytes(self) -> int:
        dim = self.embedder.get_sentence_embedding_dimension()
        return len(self._records) * dim * self.bit_width // 8


class RescoringIndex:
    """Over-fetch from a first-stage index, reorder by full-precision cosine."""

    def __init__(self, first_stage: Any, embedder: Any, candidate_multiplier: int = 5) -> None:
        self.first_stage = first_stage
        self.embedder = embedder
        self.candidate_multiplier = candidate_multiplier

    def search(
        self, query: str, top_k: int = 8, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        import numpy as np

        candidates = self.first_stage.search(
            query,
            top_k=max(top_k * self.candidate_multiplier, top_k),
            metadata_filter=metadata_filter,
        )
        if not candidates:
            return []
        q = np.asarray(self.embedder.encode(query, convert_to_numpy=True), dtype="float32")
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        mat = np.asarray(
            self.embedder.encode([c["text"] for c in candidates], convert_to_numpy=True),
            dtype="float32",
        )
        mat = mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12, None)
        sims = mat @ q
        order = np.argsort(-sims)[:top_k]
        return [{**candidates[int(i)], "score": float(sims[int(i)])} for i in order]


def turbovec_enabled() -> bool:
    return os.getenv("STORE_BACKEND", "").strip().lower() == "turbovec"
