"""Table extraction from PDFs via pdfplumber.

Financial PDFs carry their most valuable content in tables — results
statements, capital structure, maturity schedules. A plain text-layer pull
(`pdf_text`) linearises those into a stream where a figure and its label drift
apart, exactly the failure that hurt retrieval in the template. Extracting
tables explicitly and linearising each row as ``label: value`` pairs keeps every
number next to the header it belongs to, so a specific figure and its context
land in the same chunk.
"""

from __future__ import annotations

from typing import Any

from frag.sources.chunking import make_chunk_records


def _cell(value: Any) -> str:
    return "" if value is None else str(value).strip().replace("\n", " ")


def linearize_table(rows: list[list[Any]]) -> str:
    """Turn a table (list of rows) into retrieval-friendly text.

    The first non-empty row is treated as the header. Each subsequent row is
    emitted as ``h1: v1 | h2: v2 | ...`` so a value never loses its column label.
    Header-less or single-column tables fall back to pipe-joined rows.
    """
    cleaned = [[_cell(c) for c in row] for row in rows if any(_cell(c) for c in row)]
    if not cleaned:
        return ""

    header = cleaned[0]
    body = cleaned[1:]
    if not body or all(not h for h in header):
        return "\n".join(" | ".join(c for c in row if c) for row in cleaned)

    lines: list[str] = []
    for row in body:
        pairs = [f"{h}: {v}" if h else v for h, v in zip(header, row, strict=False) if v]
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)


def extract_tables(path: str) -> list[dict[str, Any]]:
    """Return every table found, as ``{page, rows}`` (1-indexed pages)."""
    import pdfplumber  # imported lazily so module import stays cheap

    tables: list[dict[str, Any]] = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for rows in page.extract_tables() or []:
                if rows:
                    tables.append({"page": page_no, "rows": rows})
    return tables


def tables_to_records(
    path: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Extract tables, linearise each, and return canonical chunk records.

    One source table can exceed `chunk_size`, so each linearised table is run
    through `make_chunk_records`; `metadata.ingest_path` is tagged `pdf_table`
    and `metadata.page` records where the table came from.
    """
    base = dict(metadata or {})
    base.setdefault("source", "pdf")
    base.setdefault("source_type", "pdf")
    base["ingest_path"] = "pdf_table"

    records: list[dict[str, Any]] = []
    for i, table in enumerate(extract_tables(path)):
        text = linearize_table(table["rows"])
        if not text:
            continue
        meta = {**base, "page": table["page"], "table_index": i}
        records.extend(
            make_chunk_records(
                text,
                metadata=meta,
                source_url=f"{base.get('source_url', path)}#table{i}",
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return records
