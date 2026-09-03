"""Pure data-prep for embedding finetune and reranker training (no GPU, no fit)."""

from __future__ import annotations

import json

from frag.train import finetune_embedding as fe
from frag.train import train_reranker as tr

_PAIRS = [
    {
        "query": "What was revenue?",
        "positive": "Revenue rose to 5,000.",
        "positive_id": "d1",
        "negatives": [
            {"id": "d2", "text": "Cash flow was strong."},
            {"id": "d3", "text": "Risks remain."},
        ],
    },
    {
        "query": "Margin trend?",
        "positive": "Operating margin expanded.",
        "positive_id": "d4",
        "negatives": [],
    },
    {"query": "", "positive": "ignored empty query", "negatives": []},
]


def test_embedding_samples_use_first_hard_negative():
    samples = fe.to_training_samples(_PAIRS)
    # empty-query pair dropped; first pair -> triple, second -> pair
    assert samples[0] == ["What was revenue?", "Revenue rose to 5,000.", "Cash flow was strong."]
    assert samples[1] == ["Margin trend?", "Operating margin expanded."]
    assert len(samples) == 2


def test_reranker_samples_label_positive_and_negatives():
    samples = tr.to_reranker_samples(_PAIRS)
    labels = {tuple(t): lbl for t, lbl in samples}
    assert labels[("What was revenue?", "Revenue rose to 5,000.")] == 1.0
    assert labels[("What was revenue?", "Cash flow was strong.")] == 0.0
    assert labels[("What was revenue?", "Risks remain.")] == 0.0
    # 1 positive + 2 negatives from pair 1, 1 positive from pair 2, empty dropped
    assert len(samples) == 4


def test_load_pairs_roundtrip(tmp_path):
    p = str(tmp_path / "pairs.jsonl")
    with open(p, "w", encoding="utf-8") as fh:
        for row in _PAIRS[:2]:
            fh.write(json.dumps(row) + "\n")
    assert len(fe.load_pairs(p)) == 2
    assert len(tr.load_pairs(p)) == 2
