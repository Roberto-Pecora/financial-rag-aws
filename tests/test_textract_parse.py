"""Textract OCR parsing and spend-gate tests — no AWS credentials required.

`parse_textract_lines` is pure, so it is tested against a recorded response
shape. The billable OCR call is tested only for its refusal-to-run guard.
"""

from __future__ import annotations

import pytest

from frag.sources import ocr_textract

# A minimal slice of a Textract GetDocumentTextDetection response.
_RESPONSE = {
    "JobStatus": "SUCCEEDED",
    "Blocks": [
        {"BlockType": "PAGE", "Page": 1},
        {"BlockType": "LINE", "Page": 1, "Text": "ANNUAL REPORT 2026"},
        {"BlockType": "LINE", "Page": 1, "Text": "Total revenue: 5,678"},
        {"BlockType": "WORD", "Page": 1, "Text": "ignored"},
        {"BlockType": "LINE", "Page": 2, "Text": "Net income: 910"},
    ],
}


def test_parse_textract_lines_orders_by_page_and_drops_non_lines():
    text = ocr_textract.parse_textract_lines(_RESPONSE)
    assert text == "ANNUAL REPORT 2026\nTotal revenue: 5,678\nNet income: 910"
    assert "ignored" not in text  # WORD blocks excluded


def test_parse_textract_lines_accepts_paginated_list():
    page1 = {"Blocks": [{"BlockType": "LINE", "Page": 1, "Text": "a"}]}
    page2 = {"Blocks": [{"BlockType": "LINE", "Page": 2, "Text": "b"}]}
    assert ocr_textract.parse_textract_lines([page1, page2]) == "a\nb"


def test_parse_textract_lines_accepts_raw_block_list():
    blocks = [{"BlockType": "LINE", "Page": 1, "Text": "hello 42"}]
    assert ocr_textract.parse_textract_lines(blocks) == "hello 42"


def test_ocr_refuses_without_confirm(monkeypatch):
    monkeypatch.setenv("TEXTRACT_ENABLED", "1")
    with pytest.raises(RuntimeError, match="billable and gated"):
        ocr_textract.ocr_pdf_in_s3("bucket", "key.pdf", confirm=False)


def test_ocr_refuses_when_disabled(monkeypatch):
    monkeypatch.setenv("TEXTRACT_ENABLED", "0")
    with pytest.raises(RuntimeError, match="billable and gated"):
        ocr_textract.ocr_pdf_in_s3("bucket", "key.pdf", confirm=True)


def test_ocr_to_records_tags_textract_path():
    records = ocr_textract.ocr_to_records(
        "Total revenue: 5,678\nNet income: 910",
        metadata={"ticker": "TSLA"},
        source_url="s3://bucket/scan.pdf",
        chunk_size=200,
        overlap=20,
    )
    assert records
    assert all(r["metadata"]["ingest_path"] == "textract" for r in records)
    assert any("5,678" in r["text"] for r in records)
