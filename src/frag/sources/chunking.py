from __future__ import annotations

import hashlib
import os
import re
from typing import Any

from bs4 import BeautifulSoup

# SEC filings are organised into Parts and numbered Items (e.g. "PART II",
# "Item 7", "ITEM 8. FINANCIAL STATEMENTS"). These are unambiguous structural
# boundaries; splitting on them keeps a section's content together instead of
# the blind fixed-window split cutting through tables and sentences. All-caps
# section titles are deliberately NOT used as boundaries - in the flattened
# text they collide with person names and mid-word fragments.
_SECTION_HEADER = re.compile(
    r"^\s*(PART\s+[IVX]+\b|Item\s+\d+[A-Z]?\b|ITEM\s+\d+[A-Z]?\b)",
    re.MULTILINE,
)


def html_to_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    if "<html" in raw_text.lower() or "<body" in raw_text.lower():
        soup = BeautifulSoup(raw_text, "html.parser")
        # iXBRL wraps visible numbers/text in <ix:nonFraction>/<ix:nonNumeric>
        # tags, so unwrap (not decompose) those to keep the real content; only
        # the hidden <ix:header> metadata block is dropped entirely.
        for tag in soup.find_all("ix:header"):
            tag.decompose()
        for tag in soup.find_all(lambda t: t.name and ":" in t.name):
            tag.replace_with(tag.get_text())
        text = soup.get_text("\n", strip=True)
    else:
        text = raw_text

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _window_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-size sliding-window split (the original strategy)."""
    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def _section_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split on SEC Part/Item headers first, then window-split any section
    still longer than chunk_size. Sections shorter than chunk_size are kept
    whole, so a results table or MD&A subsection stays in one chunk."""
    boundaries = [m.start() for m in _SECTION_HEADER.finditer(text)]
    if not boundaries:
        return _window_split(text, chunk_size, overlap)

    # Ensure we cover the preamble before the first header.
    if boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(text))

    chunks: list[str] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        section = text[start:end].strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(_window_split(section, chunk_size, overlap))
    return chunks


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if os.getenv("CHUNK_STRATEGY", "window").lower() == "section":
        return _section_split(text, chunk_size, overlap)
    return _window_split(text, chunk_size, overlap)


def make_chunk_records(
    text: str,
    metadata: dict[str, Any] | None = None,
    source_url: str | None = None,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    metadata = dict(metadata or {})
    clean_text = html_to_text(text)

    if not clean_text:
        return []

    if source_url:
        metadata["source_url"] = source_url

    chunks = chunk_text(clean_text, chunk_size=chunk_size, overlap=overlap)
    records: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        doc_id_src = (
            f"{metadata.get('ticker', '')}|{metadata.get('filing_date', '')}|"
            f"{source_url or ''}|{i}|{chunk[:80]}"
        )
        doc_id = hashlib.md5(doc_id_src.encode("utf-8")).hexdigest()

        records.append(
            {
                "id": doc_id,
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                },
            }
        )

    return records
