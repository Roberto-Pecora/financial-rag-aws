"""Ablation matrix over the retrieval stack: quality + latency + cost per cell.

Sweeps {base vs finetuned embedding} x {rerank off vs on}. Each cell reports
content-based IR metrics (mean +- sd over repeats) plus retrieval+rerank latency
(p50/p95). LLM cost is 0 for the local retrieval sweep; the end-to-end bench
measures it where an LLM is actually in the loop.
"""

from __future__ import annotations

from typing import Any

from frag.eval.harness import evaluate
from frag.eval.local_retrieval import LocalDenseIndex
from frag.eval.trace import latency_summary, mean_sd, time_call

_METRICS = ("recall@1", "recall@10", "ndcg@10")


def _timed_predictions(search, golden, top_k):
    """Run each golden query through `search`, timing it; return (preds, latencies_s)."""
    preds, latencies = [], []
    for g in golden:
        hits, elapsed = time_call(search, g["query"], top_k)
        preds.append({"query": g["query"], "retrieved_docs": hits, "critic_score": 0.0})
        latencies.append(elapsed)
    return preds, latencies


def _run_cell(search, golden, top_k, repeats):
    """Evaluate one cell `repeats` times; aggregate metric mean+-sd and latency p50/p95."""
    per_repeat: list[dict[str, float]] = []
    all_latencies: list[float] = []
    for _ in range(repeats):
        preds, latencies = _timed_predictions(search, golden, top_k)
        per_repeat.append(evaluate(preds, golden)["summary"])
        all_latencies.extend(latencies)

    metrics, metrics_sd = {}, {}
    for m in _METRICS:
        mean, sd = mean_sd([r.get(m, 0.0) for r in per_repeat])
        metrics[m], metrics_sd[m] = mean, sd
    return {"metrics": metrics, "metrics_sd": metrics_sd, "latency": latency_summary(all_latencies)}


def run_local_ablation(
    corpus: list[dict[str, Any]],
    golden: list[dict[str, Any]],
    embedders: dict[str, Any],
    rerankers: dict[str, Any] | None = None,
    top_k: int = 10,
    repeats: int = 1,
    mlflow_logger: Any | None = None,
) -> list[dict[str, Any]]:
    """Evaluate each (embedder, reranker) cell locally (no AWS, no LLM cost)."""
    from frag.rag.reranker import RerankingStore

    rerankers = rerankers or {"rerank-off": None}
    results: list[dict[str, Any]] = []
    for emb_name, embedder in embedders.items():
        index = LocalDenseIndex(corpus, embedder)
        for rr_name, reranker in rerankers.items():
            store = index if reranker is None else RerankingStore(index, reranker)
            cell = _run_cell(store.search, golden, top_k, repeats)
            cell["name"] = f"{emb_name} + {rr_name}"
            cell["cost_usd"] = 0.0  # local retrieval; LLM cost measured end-to-end
            results.append(cell)
            if mlflow_logger is not None:
                mlflow_logger(cell)
    return results


def format_markdown(results: list[dict[str, Any]], metrics: tuple[str, ...] = _METRICS) -> str:
    """Render cells as a Markdown table: metrics (mean+-sd if >0), p50/p95 ms, cost."""
    cols = list(metrics) + ["p50 ms", "p95 ms", "cost $"]
    lines = ["| Configuration | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for r in results:
        cells = []
        for m in metrics:
            sd = r.get("metrics_sd", {}).get(m, 0.0)
            val = r["metrics"][m]
            cells.append(f"{val:.3f} ± {sd:.3f}" if sd else f"{val:.3f}")
        cells.append(f"{r['latency']['p50_ms']:.0f}")
        cells.append(f"{r['latency']['p95_ms']:.0f}")
        cells.append(f"{r.get('cost_usd', 0.0):.4f}")
        lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
