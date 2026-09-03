"""Run the retrieval ablation matrix and print a Markdown findings table (no AWS).

    python scripts/run_ablation.py --corpus data/corpus.jsonl \
        --base BAAI/bge-small-en-v1.5 --finetuned artifacts/bge-ft \
        --reranker artifacts/reranker

Sweeps {base, finetuned} embeddings x {rerank off, on} over the golden set,
logs each cell to MLflow, and prints the table for the README. The dense/hybrid
axis is added by pointing at the live OpenSearch store (separate runner).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from frag.eval.ablation import format_markdown, run_local_ablation
from frag.utils.config import configure_runtime


def _load_corpus(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_golden(path):
    df = pd.read_csv(path)
    return [
        {"query": r["query"], "reference_answer": r.get("reference_answer", "")}
        for _, r in df.iterrows()
    ]


def main():
    configure_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--golden", default="data/golden_seed.csv")
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--finetuned", default=None, help="path to finetuned embedding artifact")
    ap.add_argument("--reranker", default=None, help="path to trained cross-encoder artifact")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    from frag.rag.reranker import CrossEncoderReranker

    embedders = {"base bge-small": SentenceTransformer(args.base)}
    if args.finetuned:
        embedders["finetuned bge-small"] = SentenceTransformer(args.finetuned)

    rerankers = {"rerank-off": None}
    if args.reranker:
        rerankers["rerank-on"] = CrossEncoderReranker(model_name=args.reranker)

    try:
        import mlflow

        def logger(name, summary):
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"config": name, "top_k": args.top_k})
                mlflow.log_metrics(
                    {k: float(v) for k, v in summary.items() if isinstance(v, int | float)}
                )
    except Exception:
        logger = None

    corpus = _load_corpus(args.corpus)
    golden = _load_golden(args.golden)
    results = run_local_ablation(
        corpus, golden, embedders, rerankers, top_k=args.top_k, mlflow_logger=logger
    )
    print("\n" + format_markdown(results) + "\n")


if __name__ == "__main__":
    main()
