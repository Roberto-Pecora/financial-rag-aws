from __future__ import annotations

import os
from typing import Any

import requests

from frag.sources.chunking import make_chunk_records
from frag.sources.edgar_fetch import build_headers, fetch_html
from frag.sources.sec.discovery import discover_filings

# Chunking config is env-driven so the values used at ingest time are the same
# ones logged as MLflow params at eval time (see scripts/run_eval.py), keeping
# experiment runs self-describing when the chunking strategy is varied.
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


def load_sample_sources() -> list[dict[str, Any]]:
    return [
        {
            "id": "sample_aapl_risks",
            "text": (
                "Apple's filings discuss risks including competition, supply chain "
                "constraints, regulatory pressure, and macroeconomic conditions."
            ),
            "metadata": {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "source_type": "sample",
                "form_type": "10-K",
            },
        },
        {
            "id": "sample_fred_dgs10",
            "text": (
                "The 10-year Treasury yield is a commonly used benchmark interest "
                "rate in finance and valuation work."
            ),
            "metadata": {
                "series_id": "DGS10",
                "source_type": "sample_fred",
                "provider": "FRED",
            },
        },
    ]


def load_sec_sources(
    companies: list[dict[str, Any]] | None = None,
    forms: list[str] | None = None,
    limit_per_company: int = 5,
    headers: dict[str, str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    companies = companies or [
        {"cik": "0001045810", "ticker": "NVDA", "company_name": "NVIDIA CORP"},
        {"cik": "0000320193", "ticker": "AAPL", "company_name": "Apple Inc."},
        {"cik": "0000789019", "ticker": "MSFT", "company_name": "MICROSOFT CORP"},
    ]
    forms = forms or ["10-K", "10-Q", "8-K"]
    request_headers = build_headers(headers)

    filings = discover_filings(
        companies=companies,
        forms=forms,
        limit_per_company=limit_per_company,
        headers=request_headers,
    )

    docs: list[dict[str, Any]] = []
    for filing in filings:
        url = filing.get("url")
        if not url:
            continue

        html = fetch_html(url, headers=request_headers)
        metadata = filing.get("metadata", {}) or {}

        docs.extend(
            make_chunk_records(
                text=html,
                metadata=metadata,
                source_url=url,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return docs


def load_fred_sources(
    fred_series_ids: list[str] | None = None,
    fred_api_key: str | None = None,
    limit: int = 24,
) -> list[dict[str, Any]]:
    fred_series_ids = fred_series_ids or []
    if not fred_series_ids:
        return []

    api_key = fred_api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required when fred_series_ids is provided.")

    docs: list[dict[str, Any]] = []
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    for series_id in fred_series_ids:
        response = requests.get(
            base_url,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        observations = payload.get("observations", [])

        if not observations:
            continue

        text = "\n".join(f"{obs.get('date')}: {obs.get('value')}" for obs in observations)

        docs.append(
            {
                "id": f"fred_{series_id}_{observations[0].get('date')}",
                "text": f"FRED series {series_id}\n{text}",
                "metadata": {
                    "series_id": series_id,
                    "provider": "FRED",
                    "source_type": "fred",
                },
            }
        )

    return docs


def load_live_sources(
    companies: list[dict[str, Any]] | None = None,
    forms: list[str] | None = None,
    limit_per_company: int = 5,
    headers: dict[str, str] | None = None,
    fred_series_ids: list[str] | None = None,
    fred_api_key: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    sec_docs = load_sec_sources(
        companies=companies,
        forms=forms,
        limit_per_company=limit_per_company,
        headers=headers,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    fred_docs = load_fred_sources(
        fred_series_ids=fred_series_ids,
        fred_api_key=fred_api_key,
    )

    return sec_docs + fred_docs
