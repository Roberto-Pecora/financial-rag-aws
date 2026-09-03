from __future__ import annotations

from typing import Any

from frag.sources.sec.client import archive_cik, fetch_submissions_json, zero_pad_cik


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def normalize_form(form: Any) -> str:
    if form is None:
        return ""
    value = str(form).strip().upper()
    value = value.replace("FORM ", "").replace("FORM", "")
    value = value.replace(" ", "")
    value = value.replace("-", "")
    return value


def build_filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_archive = archive_cik(cik)
    accession_no_dash = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_archive}/{accession_no_dash}/{primary_document}"
    )


def submission_to_filings(
    cik: str,
    ticker: str,
    company_name: str | None = None,
    forms: list[str] | None = None,
    limit: int = 10,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    data = fetch_submissions_json(cik, headers=headers)
    recent = data.get("filings", {}).get("recent", {}) or {}

    accession_numbers = recent.get("accessionNumber") or []
    primary_docs = recent.get("primaryDocument") or []
    filing_dates = recent.get("filingDate") or []
    forms_found = recent.get("form") or []

    allowed_forms = {normalize_form(f) for f in (forms or ["10-K", "10-Q", "8-K"])}
    cik10 = zero_pad_cik(cik)

    filings: list[dict[str, Any]] = []

    for accession, primary_doc, filing_date, form in zip(
        accession_numbers, primary_docs, filing_dates, forms_found, strict=True
    ):
        form_norm = normalize_form(form)
        if form_norm not in allowed_forms:
            continue
        if not accession or not primary_doc or not filing_date:
            continue

        filing_url = build_filing_url(
            cik=cik10, accession_number=accession, primary_document=primary_doc
        )

        filings.append(
            {
                "url": filing_url,
                "metadata": {
                    "cik": cik10,
                    "ticker": normalize_ticker(ticker),
                    "company_name": company_name or data.get("name"),
                    "form_type": form,
                    "filing_date": filing_date,
                    "accession_number": accession,
                    "primary_document": primary_doc,
                    "source_type": "sec",
                    "provider": "sec",
                },
            }
        )

        if len(filings) >= limit:
            break

    return filings
