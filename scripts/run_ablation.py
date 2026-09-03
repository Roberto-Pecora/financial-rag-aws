"""Retrieval ablation matrix with quality + latency + cost, logged to MLflow.

    python scripts/run_ablation.py --corpus data/corpus.jsonl \
        --base BAAI/bge-small-en-v1.5 --finetuned artifacts/bge-ft \
        --reranker artifacts/reranker --repeats 5
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


def _mlflow_logger(top_k, repeats):
    try:
        import mlflow
    except Exception:
        return None

    def log(cell):
        with mlflow.start_run(run_name=cell["name"]):
            mlflow.log_params({"config": cell["name"], "top_k": top_k, "repeats": repeats})
            metrics = {k: float(v) for k, v in cell["metrics"].items()}
            metrics.update({f"{k}_sd": float(v) for k, v in cell["metrics_sd"].items()})
            metrics.update({k: float(v) for k, v in cell["latency"].items()})
            metrics["cost_usd"] = float(cell["cost_usd"])
            mlflow.log_metrics(metrics)

    return log


def main():
    configure_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--golden", default="data/golden_seed.csv")
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--finetuned", default=None)
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    from frag.rag.reranker import CrossEncoderReranker

    embedders = {"base bge-small": SentenceTransformer(args.base)}
    if args.finetuned:
        embedders["finetuned bge-small"] = SentenceTransformer(args.finetuned)

    rerankers = {"rerank-off": None}
    if args.reranker:
        rerankers["rerank-on"] = CrossEncoderReranker(model_name=args.reranker)

    results = run_local_ablation(
        _load_corpus(args.corpus),
        _load_golden(args.golden),
        embedders,
        rerankers,
        top_k=args.top_k,
        repeats=args.repeats,
        mlflow_logger=_mlflow_logger(args.top_k, args.repeats),
    )
    print("\n" + format_markdown(results) + "\n")


if __name__ == "__main__":
    main()
