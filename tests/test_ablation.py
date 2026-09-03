"""Ablation: markdown formatting (pure) and a fake-embedder local sweep."""

from __future__ import annotations

import numpy as np

from frag.eval import ablation
from frag.rag.reranker import CrossEncoderReranker


def test_format_markdown_table():
    results = [
        {
            "name": "base + rerank-off",
            "summary": {"recall@1": 0.1, "recall@10": 0.5, "ndcg@10": 0.24},
        },
        {
            "name": "finetuned + rerank-on",
            "summary": {"recall@1": 0.3, "recall@10": 0.7, "ndcg@10": 0.41},
        },
    ]
    md = ablation.format_markdown(results)
    assert "| Configuration | recall@1 | recall@10 | ndcg@10 |" in md
    assert "| finetuned + rerank-on | 0.300 | 0.700 | 0.410 |" in md


class _FakeEmbedder:
    def _vec(self, t):
        t = t.lower()
        return np.array([1.0 if "revenue" in t else 0.0, 1.0 if "risk" in t else 0.0]) + 1e-6

    def encode(self, text, convert_to_numpy=True):
        return self._vec(text) if isinstance(text, str) else np.vstack([self._vec(x) for x in text])


class _FakeCE:
    def predict(self, pairs):
        return [1.0 if "revenue" in txt.lower() else 0.0 for _q, txt in pairs]


def test_run_local_ablation_produces_cells():
    corpus = [
        {"id": "rev", "text": "Total revenue increased to 5,000.", "metadata": {"doc_id": "rev"}},
        {"id": "risk", "text": "Key risk factors remain.", "metadata": {"doc_id": "risk"}},
    ]
    golden = [{"query": "what was revenue", "reference_answer": "Revenue increased to 5,000."}]
    logged = []
    results = ablation.run_local_ablation(
        corpus,
        golden,
        embedders={"base": _FakeEmbedder()},
        rerankers={"rerank-off": None, "rerank-on": CrossEncoderReranker(model=_FakeCE())},
        top_k=2,
        mlflow_logger=lambda name, summary: logged.append(name),
    )
    names = [r["name"] for r in results]
    assert names == ["base + rerank-off", "base + rerank-on"]
    assert logged == names  # logger called per cell
    assert all("summary" in r for r in results)
