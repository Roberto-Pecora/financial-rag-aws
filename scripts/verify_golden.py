"""Re-verify golden_seed.csv gold_doc_ids against the current corpus.

Chunk boundaries shift whenever chunking/ingestion changes, so a doc_id
recorded as "gold" for one ingest can miss content that moved into a
neighbouring chunk on the next. This script checks each golden row's
reference_answer for the numeric/key facts it asserts, finds every chunk
(scoped by the row's metadata_filter) whose text contains all of them, and
reports any chunk not already listed in gold_doc_ids so it can be added.

Usage: python scripts/verify_golden.py [--write]
  --write   apply the expanded gold_doc_ids back to data/golden_seed.csv
            (without it, the script only reports what it would add)
"""

from __future__ import annotations

import argparse
import ast
import re
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, "src")
from frag.rag.store_qdrant import QdrantStore  # noqa: E402

load_dotenv()

GOLDEN_PATH = "data/golden_seed.csv"


def _parse_literal(value):
    return ast.literal_eval(value) if isinstance(value, str) and value.strip() else None


def _normalize(text: str) -> str:
    """Collapse all whitespace so table-cell renderings like "$\\n155,237"
    match prose renderings like "$155,237" under substring search."""
    return re.sub(r"\s+", "", text)


def extract_facts(reference_answer: str) -> list[str]:
    """Pull numeric figures to use as match keys. Drops the $ sign, since
    table cells often split it from the number; excludes bare 4-digit years."""
    candidates = re.findall(r"\$?\d[\d,]*(?:\.\d+)?%?", reference_answer)
    facts = []
    for c in candidates:
        if re.fullmatch(r"\d{4}", c):
            continue
        digits_only = re.sub(r"[^\d]", "", c)
        if len(digits_only) >= 3 or "." in c or "%" in c:
            facts.append(_normalize(c.lstrip("$")))
    return facts


def find_matching_chunks(
    store: QdrantStore, facts: list[str], metadata_filter: dict | None
) -> list[str]:
    if not facts:
        return []
    results = store.client.scroll(
        collection_name=store.collection_name,
        scroll_filter=_build_filter(metadata_filter) if metadata_filter else None,
        limit=5000,
        with_payload=True,
    )[0]
    matches = []
    for point in results:
        text = _normalize(point.payload.get("text", ""))
        if all(fact in text for fact in facts):
            matches.append(point.payload.get("doc_id", ""))
    return matches


def _build_filter(metadata_filter: dict):
    from qdrant_client import models

    must = [
        models.FieldCondition(key=k, match=models.MatchValue(value=v))
        for k, v in metadata_filter.items()
    ]
    return models.Filter(must=must)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(GOLDEN_PATH)
    store = QdrantStore()

    all_ids = {
        point.payload.get("doc_id", "")
        for point in store.client.scroll(
            collection_name=store.collection_name, limit=5000, with_payload=True
        )[0]
    }

    changed = False
    for idx, row in df.iterrows():
        existing_ids = set(_parse_literal(row["gold_doc_ids"]) or [])
        metadata_filter = _parse_literal(row.get("metadata_filter"))
        facts = extract_facts(row["reference_answer"])

        stale_ids = existing_ids - all_ids
        found_ids = set(find_matching_chunks(store, facts, metadata_filter)) if facts else set()
        new_ids = found_ids - existing_ids
        final_ids = (existing_ids | found_ids) - stale_ids

        if stale_ids:
            print(f"[row {idx}] {row['query']!r} -- STALE gold_doc_ids (no longer in corpus):")
            print(f"  {sorted(stale_ids)}")
            changed = True
        if new_ids:
            print(f"  facts checked: {facts}")
            print(f"  new matches:   {sorted(new_ids)}")
            changed = True
        if facts and not found_ids:
            print(f"[row {idx}] {row['query']!r} -- WARNING: no chunk matched facts {facts}")
        if not final_ids:
            print(f"[row {idx}] {row['query']!r} -- WARNING: gold_doc_ids now EMPTY")
            print("  needs manual re-verification")

        if args.write and final_ids != existing_ids:
            df.at[idx, "gold_doc_ids"] = str(sorted(final_ids))

    if args.write and changed:
        df.to_csv(GOLDEN_PATH, index=False)
        print(f"\nWrote expanded gold_doc_ids to {GOLDEN_PATH}")
    elif not changed:
        print("\nNo new gold chunks found; golden set is already consistent with the corpus.")
    else:
        print("\nDry run only -- re-run with --write to apply changes.")
