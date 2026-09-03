from frag.sources.sec import (
    build_filing_url,
    build_sec_headers,
    fetch_company_tickers_json,
    normalize_form,
    submission_to_filings,
)


def test_sec_modules_expose_expected_helpers():
    headers = build_sec_headers({"User-Agent": "test-agent"})
    assert headers["User-Agent"] == "test-agent"
    assert normalize_form("Form 10-K") == "10K"
    assert build_filing_url("0000320193", "0000320193-24-000032", "aapl-20240629.htm") == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000032/aapl-20240629.htm"
    )

    assert callable(submission_to_filings)
    assert callable(fetch_company_tickers_json)
