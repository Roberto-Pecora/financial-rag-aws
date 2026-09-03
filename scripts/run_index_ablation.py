"""Index-axis ablation: exact float32 vs turbovec (4-bit) vs turbovec+rescore.

    python scripts/run_index_ablation.py --corpus data/corpus.jsonl --bit-width 4

Reports recall/nDCG, latency (p50/p95), and index memory (MB) per index, so the
quantisation trade-off is measured on our corpus rather than quoted. Needs the
[efficient] extra (turbovec). No AWS.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from frag.eval.ablation import format_index_markdown, run_index_ablation
from frag.eval.local_retrieval import LocalDenseIndex
from frag.rag.embedders import make_embedder
from frag.rag.store_turbovec import RescoringIndex, TurbovecIndex
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
    ap.add_argument("--bit-width", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    corpus = _load_corpus(args.corpus)
    golden = _load_golden(args.golden)
    embedder = make_embedder()

    turbo = TurbovecIndex(corpus, embedder, bit_width=args.bit_width)
    indexes = {
        "exact float32": LocalDenseIndex(corpus, embedder),
        f"turbovec-{args.bit_width}bit": turbo,
        f"turbovec-{args.bit_width}bit + rescore": RescoringIndex(turbo, embedder),
    }
    results = run_index_ablation(indexes, golden, top_k=args.top_k, repeats=args.repeats)
    print("\n" + format_index_markdown(results) + "\n")


if __name__ == "__main__":
    main()
