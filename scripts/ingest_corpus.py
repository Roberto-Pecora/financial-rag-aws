"""Chunk a corpus_build JSONL and ingest it into the configured store.

    python scripts/ingest_corpus.py --corpus data/corpus_build/corpus.jsonl

Whole documents (CUAD contracts are large) are chunked before indexing, so each
stored record is a retrievable passage rather than one truncated vector. Uses the
store selected by STORE_BACKEND (OpenSearch here).
"""

from __future__ import annotations

import argparse
import json
import os

from frag.rag.controller import RagController
from frag.sources.chunking import make_chunk_records
from frag.utils.config import configure_runtime


def main():
    configure_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_build/corpus.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="cap total chunks (quick demo)")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    chunk_size = int(os.getenv("CHUNK_SIZE", "600"))
    records = []
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            records.extend(
                make_chunk_records(rec.get("text", ""), metadata=rec.get("metadata", {}),
                                   chunk_size=chunk_size)
            )
    # Financial evidence first, so financial queries work before the (larger,
    # slower) legal-contract set finishes.
    records.sort(key=lambda r: 0 if r["metadata"].get("source") == "financebench" else 1)
    if args.limit:
        records = records[: args.limit]

    controller = RagController()
    total = 0
    for i in range(0, len(records), args.batch):
        total += controller.ingest(records[i : i + args.batch])
        print(f"ingested {total}/{len(records)}", flush=True)

    print({"chunks": len(records), "indexed": total, "store_count": controller.count()})


if __name__ == "__main__":
    main()
