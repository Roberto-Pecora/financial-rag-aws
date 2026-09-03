"""Score a SentenceTransformer's retrieval on the golden set, in-memory (no AWS).

    python scripts/eval_local.py --corpus data/corpus.jsonl --model BAAI/bge-small-en-v1.5
    python scripts/eval_local.py --corpus data/corpus.jsonl --model artifacts/bge-ft

Run it twice (base then finetuned) to read the finetune lift on recall@k / nDCG.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from frag.eval.local_retrieval import LocalDenseIndex, evaluate_index


def _load_corpus(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_golden(path):
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append({"query": r["query"], "reference_answer": r.get("reference_answer", "")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--golden", default="data/golden_seed.csv")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    index = LocalDenseIndex(_load_corpus(args.corpus), SentenceTransformer(args.model))
    out = evaluate_index(index, _load_golden(args.golden), top_k=args.top_k)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
