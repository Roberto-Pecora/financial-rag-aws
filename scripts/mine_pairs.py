"""Mine (query, positive, hard-negatives) training pairs from a corpus JSONL.

Reads chunk records, generates synthetic queries per passage via OpenRouter, and
mines hard negatives with a local BM25 index. Writes pairs.jsonl.

    python scripts/mine_pairs.py --corpus data/corpus.jsonl --out data/pairs.jsonl \
        --queries-per-passage 2 --n-negatives 4 --max-passages 500

Requires OPENROUTER_API_KEY (and PAIRGEN_MODEL, default = a strong mid-tier model).
No GPU, no AWS.
"""

from __future__ import annotations

import argparse
import json

from frag.train.pair_mining import (
    InMemoryRetriever,
    SyntheticQueryGenerator,
    build_training_pairs,
    write_pairs_jsonl,
)
from frag.utils.config import configure_runtime


def _load_corpus(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    configure_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="data/pairs.jsonl")
    ap.add_argument("--queries-per-passage", type=int, default=2)
    ap.add_argument("--n-negatives", type=int, default=4)
    ap.add_argument("--max-passages", type=int, default=None)
    args = ap.parse_args()

    corpus = _load_corpus(args.corpus)
    retriever = InMemoryRetriever(corpus)
    generator = SyntheticQueryGenerator()
    pairs = build_training_pairs(
        corpus,
        generator,
        retriever,
        queries_per_passage=args.queries_per_passage,
        n_negatives=args.n_negatives,
        max_passages=args.max_passages,
    )
    n = write_pairs_jsonl(pairs, args.out)
    print({"corpus": len(corpus), "pairs": n, "out": args.out})


if __name__ == "__main__":
    main()
