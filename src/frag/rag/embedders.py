"""Embedder factory: sentence-transformers (accuracy default) or fastembed (fast)."""

from __future__ import annotations

import os
from typing import Any


def make_embedder(backend: str | None = None, model_name: str | None = None) -> Any:
    """Return an object with encode() + get_sentence_embedding_dimension()."""
    backend = (backend or os.getenv("EMBEDDING_BACKEND", "sentence-transformers")).lower()
    name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    if backend == "fastembed":
        return _FastEmbedAdapter(name)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class _FastEmbedAdapter:
    """Adapt fastembed to the sentence-transformers encode() surface used here."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._dim: int | None = None

    def encode(self, text, convert_to_numpy=True):
        import numpy as np

        single = isinstance(text, str)
        vecs = list(self._model.embed([text] if single else list(text)))
        arr = np.asarray(vecs, dtype="float32")
        if self._dim is None:
            self._dim = arr.shape[1]
        return arr[0] if single else arr

    def get_sentence_embedding_dimension(self) -> int:
        if self._dim is None:
            self.encode("dimension probe")
        return int(self._dim)
