"""Build a corpus JSONL of chunk records from SEC filings (and optional sample).

    python scripts/build_corpus.py --out data/corpus.jsonl --limit-per-company 3
    python scripts/build_corpus.py --out data/corpus.jsonl --s3-key corpus/corpus.jsonl

Writes locally; with --s3-key also uploads to the data-lake bucket (S3_BUCKET).
Requires SEC_USER_AGENT for live SEC fetching.
"""

from __future__ import annotations

import argparse
import json
import os

from frag.sources.loader import load_sample_sources, load_sec_sources
from frag.utils.config import configure_runtime


def main():
    configure_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/corpus.jsonl")
    ap.add_argument("--limit-per-company", type=int, default=3)
    ap.add_argument("--sample", action="store_true", help="use only the bundled sample corpus")
    ap.add_argument("--s3-key", default=None, help="also upload to s3://$S3_BUCKET/<key>")
    args = ap.parse_args()

    if args.sample:
        docs = load_sample_sources()
    else:
        headers = (
            {"User-Agent": os.getenv("SEC_USER_AGENT")} if os.getenv("SEC_USER_AGENT") else None
        )
        docs = load_sec_sources(headers=headers, limit_per_company=args.limit_per_company)

    with open(args.out, "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print({"records": len(docs), "out": args.out})

    if args.s3_key:
        from frag.aws.s3_store import upload_jsonl

        bucket = os.environ["S3_BUCKET"]
        n = upload_jsonl(docs, bucket, args.s3_key)
        print({"uploaded": n, "s3": f"s3://{bucket}/{args.s3_key}"})


if __name__ == "__main__":
    main()
