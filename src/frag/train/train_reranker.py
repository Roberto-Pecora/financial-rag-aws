"""Train a cross-encoder reranker on mined pairs.

The reranker scores a (query, passage) pair directly, so it sees the two texts
jointly and can catch relevance a bi-encoder misses. Training data is the same
mined pairs relabelled: the positive passage is a 1, each hard negative a 0.
`to_reranker_samples` is pure and unit-tested; `train` imports sentence-
transformers lazily and runs on a Colab GPU.
"""

from __future__ import annotations

import json
from typing import Any

DEFAULT_BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_pairs(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_reranker_samples(pairs: list[dict[str, Any]]) -> list[tuple[list[str], float]]:
    """Turn mined pairs into ([query, passage], label) with 1.0 positive / 0.0 negative."""
    samples: list[tuple[list[str], float]] = []
    for p in pairs:
        query, positive = p.get("query", ""), p.get("positive", "")
        if not query or not positive:
            continue
        samples.append(([query, positive], 1.0))
        for neg in p.get("negatives") or []:
            text = neg.get("text", "")
            if text:
                samples.append(([query, text], 0.0))
    return samples


def train(
    pairs: list[dict[str, Any]],
    out_dir: str,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 1,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
) -> str:
    """Finetune and save the cross-encoder. Heavy; runs on a GPU. Returns `out_dir`."""
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    samples = to_reranker_samples(pairs)
    examples = [InputExample(texts=texts, label=label) for texts, label in samples]
    model = CrossEncoder(base_model, num_labels=1)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    warmup = int(len(loader) * epochs * warmup_ratio)
    model.fit(train_dataloader=loader, epochs=epochs, warmup_steps=warmup)
    model.save(out_dir)
    return out_dir
