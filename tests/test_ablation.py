"""Ablation: markdown table (metrics/latency/cost) and a fake-embedder sweep with repeats."""

from __future__ import annotations

import numpy as np

from frag.eval import ablation
from frag.rag.reranker import CrossEncoderReranker


def test_format_markdown_includes_latency_and_cost():
    results = [
        {
            "name": "finetuned + rerank-on",
            "metrics": {"recall@1": 0.3, "recall@10": 0.7, "ndcg@10": 0.41},
            "metrics_sd": {"recall@1": 0.02, "recall@10": 0.0, "ndcg@10": 0.0},
            "latency": {"p50_ms": 12.0, "p95_ms": 25.0},
            "cost_usd": 0.0,
        }
    ]
    md = ablation.format_markdown(results)
    assert "p50 ms" in md and "p95 ms" in md and "cost $" in md
    assert "0.300 ± 0.020" in md  # sd shown when > 0
    assert "0.700" in md and "± 0.000" not in md.split("0.700")[1][:8]  # sd hidden when 0


class _FakeEmbedder:
    def _vec(self, t):
        t = t.lower()
        return np.array([1.0 if "revenue" in t else 0.0, 1.0 if "risk" in t else 0.0]) + 1e-6

    def encode(self, text, convert_to_numpy=True):
        return self._vec(text) if isinstance(text, str) else np.vstack([self._vec(x) for x in text])


class _FakeCE:
    def predict(self, pairs):
        return [1.0 if "revenue" in txt.lower() else 0.0 for _q, txt in pairs]


def test_run_local_ablation_cells_have_latency_and_repeats():
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
        repeats=3,
        mlflow_logger=lambda cell: logged.append(cell["name"]),
    )
    assert [r["name"] for r in results] == ["base + rerank-off", "base + rerank-on"]
    assert all("p50_ms" in r["latency"] and "p95_ms" in r["latency"] for r in results)
    assert all("metrics" in r and "metrics_sd" in r for r in results)
    assert logged == ["base + rerank-off", "base + rerank-on"]
