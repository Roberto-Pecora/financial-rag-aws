"""OCR ingestion for scanned/image-only PDFs via AWS Textract.

When a PDF has no usable text layer (`pdf_text.has_text_layer` is False), the
numbers live in pixels and only OCR recovers them. Textract's *asynchronous*
API is used because it handles multi-page PDFs directly from S3, unlike the
synchronous single-image `detect_document_text`.

Cost discipline is explicit, not incidental:
  * Textract is billable (Free Tier: 1,000 pages/month for 3 months), so an OCR
    run never starts without `confirm=True` AND `TEXTRACT_ENABLED=1`.
  * `TEXTRACT_MAX_PAGES` (default 5) caps how many pages a single call will OCR,
    so a mis-pointed large scan cannot quietly burn the quota.

`parse_textract_lines` is a pure function over Textract's response shape, so the
parsing is unit-tested with a recorded response and needs no AWS credentials.
"""

from __future__ import annotations

import os
import time
from typing import Any

from frag.sources.chunking import make_chunk_records


def parse_textract_lines(responses: Any) -> str:
    """Join the LINE blocks of one or more Textract responses into text.

    Accepts a single `GetDocumentTextDetection` response dict, a list of such
    dicts (the paginated pages of one job), or a raw list of blocks. Textract
    returns blocks in reading order; lines are grouped by `Page` so multi-page
    output stays in document order.
    """
    if isinstance(responses, dict):
        pages = [responses]
    elif responses and isinstance(responses[0], dict) and "BlockType" in responses[0]:
        pages = [{"Blocks": responses}]  # raw block list
    else:
        pages = list(responses or [])

    lines: list[tuple[int, str]] = []
    for resp in pages:
        for block in resp.get("Blocks", []):
            if block.get("BlockType") == "LINE" and block.get("Text"):
                lines.append((int(block.get("Page", 1)), block["Text"]))

    lines.sort(key=lambda pl: pl[0])
    return "\n".join(text for _, text in lines).strip()


def _enabled() -> bool:
    return os.getenv("TEXTRACT_ENABLED", "0").strip().lower() in {"1", "on", "true", "yes"}


def ocr_pdf_in_s3(
    bucket: str,
    key: str,
    confirm: bool = False,
    poll_seconds: float = 3.0,
    timeout_seconds: float = 300.0,
    profile_name: str | None = None,
    region_name: str | None = None,
) -> str:
    """OCR a PDF already stored in S3 and return its recovered text.

    Billable. Refuses to run unless `confirm=True` and `TEXTRACT_ENABLED=1`, so
    no call is made by accident. The document is capped at `TEXTRACT_MAX_PAGES`.
    """
    if not (confirm and _enabled()):
        raise RuntimeError(
            "Textract OCR is billable and gated: pass confirm=True and set "
            "TEXTRACT_ENABLED=1 to run it."
        )

    import boto3  # imported lazily so module import stays cheap and offline

    session = boto3.Session(
        profile_name=profile_name or os.getenv("AWS_PROFILE"),
        region_name=region_name or os.getenv("AWS_REGION"),
    )
    client = session.client("textract")

    start = client.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    job_id = start["JobId"]

    deadline = time.monotonic() + timeout_seconds
    while True:
        resp = client.get_document_text_detection(JobId=job_id)
        status = resp["JobStatus"]
        if status == "SUCCEEDED":
            break
        if status == "FAILED":
            raise RuntimeError(f"Textract job {job_id} failed: {resp.get('StatusMessage')}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Textract job {job_id} did not finish in {timeout_seconds}s")
        time.sleep(poll_seconds)

    max_pages = int(os.getenv("TEXTRACT_MAX_PAGES", "5"))
    pages = [resp]
    next_token = resp.get("NextToken")
    while next_token and _pages_seen(pages) < max_pages:
        resp = client.get_document_text_detection(JobId=job_id, NextToken=next_token)
        pages.append(resp)
        next_token = resp.get("NextToken")

    return parse_textract_lines(pages)


def _pages_seen(responses: list[dict[str, Any]]) -> int:
    return len({b.get("Page", 1) for r in responses for b in r.get("Blocks", [])})


def ocr_to_records(
    text: str,
    metadata: dict[str, Any] | None = None,
    source_url: str | None = None,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Turn OCR'd text into canonical chunk records tagged `ingest_path=textract`.

    Kept separate from the AWS call so a caller can OCR once, cache the text, and
    re-chunk under different strategies without paying for OCR again.
    """
    meta = dict(metadata or {})
    meta.setdefault("source", "pdf")
    meta.setdefault("source_type", "pdf")
    meta["ingest_path"] = "textract"
    return make_chunk_records(
        text,
        metadata=meta,
        source_url=source_url,
        chunk_size=chunk_size,
        overlap=overlap,
    )
