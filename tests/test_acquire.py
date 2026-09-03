"""Corpus acquisition: pure shapers (fake rows) + scanned-PDF OCR routing."""

from __future__ import annotations

from frag.sources import acquire, pdf_text


def test_cuad_records_dedupe_and_tag():
    rows = [
        {"title": "Loan Agreement", "context": "Borrower shall not incur debt above 3.0x."},
        {"title": "Loan Agreement", "context": "Borrower shall not incur debt above 3.0x."},  # dup
        {"title": "Services", "context": "The parties agree to arbitration."},
    ]
    recs = acquire.cuad_records(rows)
    assert len(recs) == 2  # duplicate context collapsed
    assert all(r["metadata"]["source"] == "cuad" for r in recs)
    assert all(r["metadata"]["ingest_path"] == "pdf_text" for r in recs)


def test_cuad_limit():
    rows = [{"context": f"contract {i}"} for i in range(10)]
    assert len(acquire.cuad_records(rows, limit=4)) == 4


def test_financebench_records_and_golden():
    rows = [
        {
            "question": "What was revenue?",
            "answer": "5,678",
            "company": "Acme",
            "evidence": [
                {
                    "evidence_text": "Revenue was 5,678.",
                    "doc_name": "ACME_10K",
                    "evidence_page_num": 42,
                }
            ],
        },
        {"question": "No evidence?", "answer": "n/a", "evidence": []},
    ]
    recs = acquire.financebench_records(rows)
    assert len(recs) == 1 and recs[0]["metadata"]["source"] == "financebench"
    assert recs[0]["metadata"]["company"] == "Acme" and recs[0]["metadata"]["page"] == 42

    golden = acquire.financebench_golden(rows)
    assert len(golden) == 2
    assert golden[0]["query"] == "What was revenue?" and golden[0]["reference_answer"] == "5,678"


def test_scanned_pdf_has_no_text_layer(tmp_path):
    """A scanned fixture must route to OCR: has_text_layer -> False."""
    p = str(tmp_path / "scan.pdf")
    acquire.make_scanned_pdf("Total revenue: 5,678\nNet income: 910", p)
    assert pdf_text.has_text_layer(p) is False  # image-only -> needs Textract
