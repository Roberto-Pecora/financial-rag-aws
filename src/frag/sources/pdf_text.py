"""Text-layer PDF ingestion via PyMuPDF.

Most financial PDFs that reach a research desk — earnings decks, investor
presentations, 10-K exports — carry an embedded text layer. This module pulls
that text out and normalises it into the canonical chunk-record schema
(`{id, text, metadata}`) via `make_chunk_records`, so a PDF flows through the
exact same retrieval/eval pipeline as the SEC/FRED sources.

Extraction stays deliberately faithful: page text is concatenated with a form
feed between pages and whitespace is only lightly normalised. The iXBRL lesson
from the template — that over-eager cleaning silently deletes the numbers — also
applies here, so nothing that looks numeric is stripped.
"""

from __future__ import annotations

import re
from typing import Any

from frag.sources.chunking import make_chunk_records

# A page break marker kept in the joined text so section/window chunking can see
# page boundaries; downstream chunkers treat it as ordinary whitespace.
_PAGE_SEP = "\n\f\n"


def _normalise(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_texts(path: str) -> list[str]:
    """Return the extracted text of each page, in order."""
    import fitz  # PyMuPDF; imported lazily so module import stays cheap

    pages: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return pages


def extract_text(path: str) -> str:
    """Return the whole document's text-layer content, page-joined."""
    return _normalise(_PAGE_SEP.join(page_texts(path)))


def has_text_layer(path: str, min_chars: int = 100) -> bool:
    """Heuristic: does this PDF carry a usable embedded text layer?

    A scanned/image-only PDF returns almost no extractable text, which is the
    signal to fall back to OCR (`ocr_textract`). The threshold is per-document,
    not per-page, so a mostly-image deck with one text page still counts.
    """
    return len(extract_text(path).strip()) >= min_chars


def pdf_to_records(
    path: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Extract text-layer content and return canonical chunk records.

    `metadata.ingest_path` is tagged `pdf_text` so per-path retrieval quality
    (and failure modes) can be measured separately from the API-clean sources.
    """
    meta = dict(metadata or {})
    meta.setdefault("source", "pdf")
    meta.setdefault("source_type", "pdf")
    meta["ingest_path"] = "pdf_text"
    return make_chunk_records(
        extract_text(path),
        metadata=meta,
        source_url=meta.get("source_url", path),
        chunk_size=chunk_size,
        overlap=overlap,
    )
