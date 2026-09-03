"""Ablation matrix over the retrieval stack.

Sweeps the IR configurations that matter for this project and scores each with
the content-based harness, so the finetune and the reranker are credited (or
not) with a like-for-like number:

    {base vs finetuned embedding} x {rerank off vs on}

The dense/hybrid axis is added when the runner targets the live OpenSearch store
(hybrid needs the server-side BM25). Here the local, AWS-free path covers the
embedding and rerank axes with the in-memory dense index.

`format_markdown` is pure and unit-tested; `run_local_ablation` loads models and
is exercised with a fake embedder.
"""

from __future__ import annotations

from typing import Any

from frag.eval.local_retrieval import LocalDenseIndex, evaluate_index

_METRICS = ("recall@1", "recall@10", "ndcg@10")


def format_markdown(results: list[dict[str, Any]], metrics: tuple[str, ...] = _METRICS) -> str:
    """Render ablation cells as a Markdown table, most-relevant metrics first."""
    header = "| Configuration | " + " | ".join(metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)
    lines = [header, sep]
    for r in results:
        cells = [f"{r['summary'].get(m, 0.0):.3f}" for m in metrics]
        lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_local_ablation(
    corpus: list[dict[str, Any]],
    golden: list[dict[str, Any]],
    embedders: dict[str, Any],
    rerankers: dict[str, Any] | None = None,
    top_k: int = 10,
    mlflow_logger: Any | None = None,
) -> list[dict[str, Any]]:
    """Evaluate each (embedder, reranker) cell; optionally log to MLflow.

    `embedders` maps a label -> a loaded SentenceTransformer-like object.
    `rerankers` maps a label -> a CrossEncoderReranker (or None for the off cell).
    """
    from frag.rag.reranker import RerankingStore

    rerankers = rerankers or {"rerank-off": None}
    results: list[dict[str, Any]] = []
    for emb_name, embedder in embedders.items():
        index = LocalDenseIndex(corpus, embedder)
        for rr_name, reranker in rerankers.items():
            name = f"{emb_name} + {rr_name}"
            if reranker is None:
                out = evaluate_index(index, golden, top_k=top_k)
            else:
                store = RerankingStore(index, reranker)
                preds = [
                    {
                        "query": g["query"],
                        "retrieved_docs": store.search(g["query"], top_k=top_k),
                        "critic_score": 0.0,
                    }
                    for g in golden
                ]
                from frag.eval.harness import evaluate

                out = evaluate(preds, golden)
            cell = {"name": name, "summary": out["summary"]}
            results.append(cell)
            if mlflow_logger is not None:
                mlflow_logger(name, out["summary"])
    return results
