"""Acquire a sizeable investment/legal corpus: CUAD + FinanceBench + live SEC (+ scan).

    python scripts/acquire_corpus.py --out-dir data/corpus_build \
        --sources cuad,financebench,sec --cuad-limit 300 --sec-limit-per-company 3

Writes corpus.jsonl (all text records), golden_financebench.csv (a real golden set),
and an optional scanned OCR fixture. CUAD/FinanceBench need the [data] extra; SEC needs
SEC_USER_AGENT. No AWS.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from frag.sources import acquire
from frag.utils.config import configure_runtime


def main():
    configure_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/corpus_build")
    ap.add_argument("--sources", default="cuad,financebench,sec")
    ap.add_argument("--cuad-limit", type=int, default=300)
    ap.add_argument("--sec-limit-per-company", type=int, default=3)
    ap.add_argument("--scanned-fixture", action="store_true", help="also emit a scanned OCR PDF")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    records: list[dict] = []
    golden: list[dict] = []

    if "cuad" in sources:
        recs = acquire.load_cuad(limit=args.cuad_limit)
        records += recs
        print({"cuad_records": len(recs)})

    if "financebench" in sources:
        fb_recs, fb_golden = acquire.load_financebench()
        records += fb_recs
        golden += fb_golden
        print({"financebench_records": len(fb_recs), "financebench_golden": len(fb_golden)})

    if "sec" in sources:
        from frag.sources.loader import load_sec_sources

        headers = (
            {"User-Agent": os.getenv("SEC_USER_AGENT")} if os.getenv("SEC_USER_AGENT") else None
        )
        sec = load_sec_sources(headers=headers, limit_per_company=args.sec_limit_per_company)
        records += sec
        print({"sec_records": len(sec)})

    corpus_path = out / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    if golden:
        gpath = out / "golden_financebench.csv"
        with open(gpath, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(golden[0].keys()))
            w.writeheader()
            w.writerows(golden)

    if args.scanned_fixture:
        acquire.make_scanned_pdf(
            "SCANNED ANNUAL REPORT 2026\nTotal revenue: 5,678\nNet income: 910",
            str(out / "raw" / "scan_fixture.pdf"),
        )

    print({"total_records": len(records), "out": str(corpus_path)})


if __name__ == "__main__":
    main()
