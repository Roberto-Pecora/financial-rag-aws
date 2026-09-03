"""Corpus acquisition: shape CUAD, FinanceBench and a scanned OCR fixture into records.

The dataset loaders are injectable (row iterables), so the shaping/golden-derivation
logic is tested with fake rows and no downloads. Loaders using HuggingFace `datasets`
are lazy and gated behind the [data] extra.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

Record = dict[str, Any]


def _doc_id(prefix: str, text: str) -> str:
    return f"{prefix}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:12]}"


def cuad_records(rows: Iterable[dict], limit: int | None = None) -> list[Record]:
    """CUAD rows -> canonical docs. One record per distinct contract (context)."""
    seen: set[str] = set()
    out: list[Record] = []
    for row in rows:
        text = (row.get("context") or "").strip()
        title = (row.get("title") or "contract").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(
            {
                "id": _doc_id("cuad", text),
                "text": text,
                "metadata": {
                    "source": "cuad",
                    "source_type": "legal_contract",
                    "ingest_path": "pdf_text",
                    "title": title,
                },
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def financebench_records(rows: Iterable[dict]) -> list[Record]:
    """FinanceBench rows -> evidence docs. `evidence` is a list of passages per row."""
    out: list[Record] = []
    seen: set[str] = set()
    for row in rows:
        for ev in row.get("evidence") or []:
            text = (ev.get("evidence_text") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(
                {
                    "id": _doc_id("fb", text),
                    "text": text,
                    "metadata": {
                        "source": "financebench",
                        "source_type": "10k",
                        "ingest_path": "pdf_text",
                        "company": row.get("company", ""),
                        "doc_name": ev.get("doc_name") or row.get("doc_name", ""),
                        "page": ev.get("evidence_page_num"),
                        "doc_period": row.get("doc_period"),
                    },
                }
            )
    return out


def financebench_golden(rows: Iterable[dict]) -> list[dict[str, Any]]:
    """FinanceBench rows -> a golden set: {query, reference_answer, metadata_filter}."""
    golden: list[dict[str, Any]] = []
    for row in rows:
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if q and a:
            golden.append(
                {
                    "query": q,
                    "gold_doc_ids": "[]",
                    "reference_answer": a,
                    "task_type": row.get("question_type", "factual"),
                    "risk_level": "medium",
                    "notes": f"financebench:{row.get('company', '')}",
                    "metadata_filter": "{}",
                }
            )
    return golden


def make_scanned_pdf(text: str, path: str) -> str:
    """Render text to an image-only PDF (no text layer) to exercise the OCR path."""
    import fitz

    src = fitz.open()
    page = src.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 520, 720), text, fontsize=12)
    pix = page.get_pixmap(dpi=150)  # rasterise -> the text becomes pixels
    src.close()

    out = fitz.open()
    img_page = out.new_page(width=pix.width, height=pix.height)
    img_page.insert_image(img_page.rect, pixmap=pix)
    out.save(path)
    out.close()
    return path


# -- lazy HuggingFace loaders (need the [data] extra + network) -------------


def load_cuad(limit: int | None = None) -> list[Record]:
    from datasets import load_dataset

    ds = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    return cuad_records(ds, limit=limit)


def load_financebench() -> tuple[list[Record], list[dict[str, Any]]]:
    from datasets import load_dataset

    ds = load_dataset("PatronusAI/financebench", split="train")
    rows = list(ds)
    return financebench_records(rows), financebench_golden(rows)
