"""Finetune bge-small on mined pairs with MultipleNegativesRankingLoss.

The data preparation (`to_training_samples`) is pure and unit-tested; the actual
fit (`train`) imports sentence-transformers lazily and is meant to run on a
Colab GPU, writing the finetuned model to `out_dir` (then uploaded to S3).

MultipleNegativesRankingLoss treats every other positive in a batch as an
in-batch negative and, when a third text is supplied, an explicit hard negative
too. Mined hard negatives (lexically-close distractors) are exactly the signal
bge-small under-discriminates on this corpus, so feeding them as the third text
targets the failure directly.
"""

from __future__ import annotations

import json
from typing import Any

DEFAULT_BASE_MODEL = "BAAI/bge-small-en-v1.5"


def load_pairs(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_training_samples(pairs: list[dict[str, Any]]) -> list[list[str]]:
    """Turn mined pairs into [query, positive, hard_negative] triples.

    A pair with no mined negative falls back to a [query, positive] pair, which
    MNRL still trains on using in-batch negatives only.
    """
    samples: list[list[str]] = []
    for p in pairs:
        query, positive = p.get("query", ""), p.get("positive", "")
        if not query or not positive:
            continue
        negatives = p.get("negatives") or []
        if negatives:
            samples.append([query, positive, negatives[0].get("text", "")])
        else:
            samples.append([query, positive])
    return samples


def train(
    pairs: list[dict[str, Any]],
    out_dir: str,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 1,
    batch_size: int = 32,
    warmup_ratio: float = 0.1,
) -> str:
    """Finetune and save the model. Heavy; runs on a GPU. Returns `out_dir`."""
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    samples = to_training_samples(pairs)
    examples = [InputExample(texts=s) for s in samples]
    model = SentenceTransformer(base_model)
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(len(loader) * epochs * warmup_ratio)
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        show_progress_bar=True,
    )
    model.save(out_dir)
    return out_dir
