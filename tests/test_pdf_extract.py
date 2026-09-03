"""Text-layer and table PDF ingestion tests.

The text-PDF fixture is generated at run time with PyMuPDF, so the suite needs
no binary fixture checked in and stays hermetic. Table extraction's core logic
(`linearize_table`) is pure and tested directly; `tables_to_records` is tested
with a stubbed extractor so it needs no ruled-table PDF.
"""

from __future__ import annotations

import fitz

from frag.sources import pdf_tables, pdf_text

_TEXT = (
    "Revenue increased to 1,234 in fiscal 2026 while liquidity remained strong. "
    "Operating margin expanded on disciplined cost control, and the company "
    "maintained a conservative capital structure with ample headroom."
)


def _make_text_pdf(path: str, body: str = _TEXT) -> str:
    doc = fitz.open()
    page = doc.new_page()
    # insert_textbox wraps within the rect, so long text stays on the page and
    # is fully recoverable by get_text (a single insert_text line overflows and
    # clips at the page edge).
    page.insert_textbox(fitz.Rect(72, 72, 520, 720), body, fontsize=12)
    doc.save(path)
    doc.close()
    return path


# --------------------------------------------------------------------------
# pdf_text
# --------------------------------------------------------------------------


def test_extract_text_recovers_numbers(tmp_path):
    pdf = _make_text_pdf(str(tmp_path / "doc.pdf"))
    text = pdf_text.extract_text(pdf)
    assert "1,234" in text  # the number survives extraction (the iXBRL lesson)
    assert "liquidity remained strong" in text


def test_has_text_layer_true_for_text_pdf(tmp_path):
    pdf = _make_text_pdf(str(tmp_path / "doc.pdf"))
    assert pdf_text.has_text_layer(pdf) is True


def test_has_text_layer_false_for_blank_pdf(tmp_path):
    blank = str(tmp_path / "blank.pdf")
    doc = fitz.open()
    doc.new_page()
    doc.save(blank)
    doc.close()
    assert pdf_text.has_text_layer(blank) is False


def test_pdf_to_records_tags_ingest_path(tmp_path):
    pdf = _make_text_pdf(str(tmp_path / "doc.pdf"))
    records = pdf_text.pdf_to_records(pdf, metadata={"ticker": "AAPL"}, chunk_size=200, overlap=20)
    assert records
    assert all(r["metadata"]["ingest_path"] == "pdf_text" for r in records)
    assert all(r["metadata"]["ticker"] == "AAPL" for r in records)
    assert all(r["id"] for r in records)  # content-hash ids present (idempotent)


def test_pdf_to_records_is_idempotent(tmp_path):
    pdf = _make_text_pdf(str(tmp_path / "doc.pdf"))
    a = pdf_text.pdf_to_records(pdf)
    b = pdf_text.pdf_to_records(pdf)
    assert [r["id"] for r in a] == [r["id"] for r in b]


# --------------------------------------------------------------------------
# pdf_tables
# --------------------------------------------------------------------------


def test_linearize_table_pairs_values_with_headers():
    rows = [
        ["Metric", "FY2025", "FY2026"],
        ["Revenue", "1,000", "1,234"],
        ["Operating income", "200", "260"],
    ]
    out = pdf_tables.linearize_table(rows)
    assert "Metric: Revenue | FY2025: 1,000 | FY2026: 1,234" in out
    assert "Operating income" in out and "260" in out


def test_linearize_table_handles_headerless():
    rows = [["10-year Treasury", "4.2%"]]
    out = pdf_tables.linearize_table(rows)
    assert "10-year Treasury" in out and "4.2%" in out


def test_linearize_table_empty():
    assert pdf_tables.linearize_table([]) == ""
    assert pdf_tables.linearize_table([[None, None]]) == ""


def test_tables_to_records_tags_page_and_path(monkeypatch):
    fake = [
        {"page": 3, "rows": [["Metric", "FY2026"], ["Revenue", "1,234"]]},
    ]
    monkeypatch.setattr(pdf_tables, "extract_tables", lambda path: fake)

    records = pdf_tables.tables_to_records("ignored.pdf", metadata={"ticker": "MSFT"})
    assert records
    assert all(r["metadata"]["ingest_path"] == "pdf_table" for r in records)
    assert records[0]["metadata"]["page"] == 3
    assert any("1,234" in r["text"] for r in records)
