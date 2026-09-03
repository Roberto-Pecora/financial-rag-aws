"""Manifest-driven, idempotent corpus ingestion.

A *manifest* is a list of source entries; each entry names a file or URL, an
ingest path, and metadata. `run_manifest` dispatches each entry to the right
extractor (`pdf_text`, `pdf_table`, `textract`, SEC HTML), collects canonical
chunk records, deduplicates them by content-hash id, and writes JSONL.

Two properties make this safe to re-run and to scale out later:
  * **Idempotent** — chunk ids are content hashes (see `make_chunk_records`), so
    re-ingesting the same document yields the same ids; a second run over an
    unchanged manifest produces byte-identical output and no duplicates.
  * **Stateless per entry** — nothing shared between entries, so the same loop
    parallelises across workers or S3 keys without coordination.

The sink is a callable so the same job writes to a local file now and to S3 in
Phase 2 without touching this logic.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from frag.sources import pdf_tables, pdf_text
from frag.sources.chunking import make_chunk_records

Record = dict[str, Any]
Entry = dict[str, Any]


def dispatch(entry: Entry) -> list[Record]:
    """Route one manifest entry to its extractor and return chunk records.

    `ingest_path` selects the extractor. For `pdf` (auto), a text layer routes to
    `pdf_text`; a scan would route to `textract`, which is billable and therefore
    skipped here unless a caller has already OCR'd the text and passes it under
    `entry["text"]` with `ingest_path="textract"`.
    """
    path = entry.get("path") or entry.get("url")
    meta = dict(entry.get("metadata") or {})
    ingest_path = entry.get("ingest_path", "auto")
    chunk_size = int(entry.get("chunk_size", 1500))
    overlap = int(entry.get("overlap", 200))

    if ingest_path in {"pdf_text", "auto"} and path and pdf_text.has_text_layer(path):
        return pdf_text.pdf_to_records(path, meta, chunk_size, overlap)
    if ingest_path == "pdf_table":
        return pdf_tables.tables_to_records(path, meta, chunk_size, overlap)
    if ingest_path == "textract":
        # OCR text must be supplied by the caller (billable step done out of band).
        text = entry.get("text", "")
        meta["ingest_path"] = "textract"
        return make_chunk_records(text, meta, path, chunk_size, overlap)
    if ingest_path in {"sec_html", "html", "ixhtml"}:
        text = entry.get("text", "")
        meta.setdefault("source", "sec")
        meta["ingest_path"] = "ixhtml"
        return make_chunk_records(text, meta, path, chunk_size, overlap)
    return []


def run_manifest(
    entries: Iterable[Entry],
    sink: Callable[[Iterable[Record]], int] | None = None,
) -> dict[str, Any]:
    """Ingest every entry, dedupe records by id, and hand them to `sink`.

    Returns stats: total entries, unique records, and a per-ingest-path count so
    retrieval quality can later be attributed to each source path.
    """
    seen: set[str] = set()
    records: list[Record] = []
    per_path: dict[str, int] = {}
    errors: list[dict[str, str]] = []

    entry_count = 0
    for entry in entries:
        entry_count += 1
        try:
            produced = dispatch(entry)
        except Exception as exc:  # a bad source must not abort the whole batch
            errors.append({"entry": str(entry.get("path") or entry.get("url")), "error": str(exc)})
            continue
        for rec in produced:
            rid = rec["id"]
            if rid in seen:
                continue  # idempotent: identical content collapses to one record
            seen.add(rid)
            records.append(rec)
            p = rec["metadata"].get("ingest_path", "unknown")
            per_path[p] = per_path.get(p, 0) + 1

    written = sink(records) if sink else 0
    return {
        "entries": entry_count,
        "records": len(records),
        "written": written,
        "per_ingest_path": per_path,
        "errors": errors,
    }


def jsonl_file_sink(path: str) -> Callable[[Iterable[Record]], int]:
    """A sink that writes records as JSONL to a local path (Phase-2: swap for S3)."""

    def _sink(records: Iterable[Record]) -> int:
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        return n

    return _sink


def load_manifest(path: str) -> list[Entry]:
    """Read a JSONL manifest (one entry object per line)."""
    entries: list[Entry] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
