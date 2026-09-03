from __future__ import annotations

import logging
from typing import Any

from frag.sources.sec.client import fetch_company_tickers_json, zero_pad_cik
from frag.sources.sec.filings import normalize_ticker, submission_to_filings

logger = logging.getLogger(__name__)


def resolve_company_identity(
    ticker: str,
    headers: dict[str, str] | None = None,
    company_tickers_json: dict[str, Any] | None = None,
) -> dict[str, str]:
    ticker_norm = normalize_ticker(ticker)
    data = company_tickers_json or fetch_company_tickers_json(headers=headers)

    for _, record in data.items():
        rec_ticker = normalize_ticker(record.get("ticker", ""))
        if rec_ticker == ticker_norm:
            cik_str = str(record["cik_str"])
            return {
                "ticker": rec_ticker,
                "cik": zero_pad_cik(cik_str),
                "company_name": record.get("title", ""),
            }

    raise ValueError(f"Ticker not found in SEC company_tickers.json: {ticker}")


def discover_filings(
    companies: list[dict[str, Any]],
    forms: list[str] | None = None,
    limit_per_company: int = 10,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    all_filings: list[dict[str, Any]] = []
    company_tickers_json = fetch_company_tickers_json(headers=headers)

    for company in companies:
        ticker = normalize_ticker(company["ticker"])

        try:
            resolved = resolve_company_identity(
                ticker=ticker,
                headers=headers,
                company_tickers_json=company_tickers_json,
            )
        except Exception as e:
            logger.warning("Skipping %s: company identity lookup failed: %s", ticker, e)
            continue

        provided_cik = company.get("cik")
        if provided_cik:
            provided_cik10 = zero_pad_cik(provided_cik)
            if provided_cik10 != resolved["cik"]:
                logger.warning(
                    "Overriding provided CIK %s with SEC mapping %s for %s",
                    provided_cik10,
                    resolved["cik"],
                    ticker,
                )

        try:
            company_filings = submission_to_filings(
                cik=resolved["cik"],
                ticker=ticker,
                company_name=company.get("company_name") or resolved["company_name"],
                forms=forms,
                limit=limit_per_company,
                headers=headers,
            )
        except Exception as e:
            logger.warning(
                "Skipping %s (%s): filing discovery failed: %s",
                ticker,
                resolved["cik"],
                e,
            )
            continue

        all_filings.extend(company_filings)

    return all_filings
