from typing import Any

import requests


def fetch_series(series_id: str, api_key: str, limit: int = 100) -> dict[str, Any]:
    """
    Low-level FRED client: fetch observations for a given series_id.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc",  # newest first
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fred_to_docs(
    series_id: str,
    api_key: str,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """
    Convert a FRED series into one or a few RAG documents.

    For now we summarize the latest value and its change vs the previous observation.
    You can extend this to include multi-period trends.
    """
    data = fetch_series(series_id, api_key, limit=limit)
    obs = data.get("observations", [])
    if not obs:
        return []

    latest = obs[0]
    latest_val = latest.get("value")
    latest_date = latest.get("date")

    if len(obs) > 1:
        prev = obs[1]
        prev_val = prev.get("value")
        prev_date = prev.get("date")
        change_text = f"changed from {prev_val} on {prev_date} to {latest_val} on {latest_date}"
    else:
        change_text = f"is {latest_val} as of {latest_date}"

    text = (
        f"The FRED series {series_id} {change_text}. "
        "Values are as reported by the Federal Reserve Economic Data (FRED) API."
    )

    return [
        {
            "doc_id": f"fred_{series_id}_{latest_date}",
            "source": "fred",
            "text": text,
            "metadata": {
                "series_id": series_id,
                "latest_date": latest_date,
                "units": data.get("units"),
                "frequency": data.get("frequency"),
            },
        }
    ]
