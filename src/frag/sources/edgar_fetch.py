from __future__ import annotations

import requests

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": "actor-critic-financial-rag/0.1 (research use; contact: local@localhost)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}


def build_headers(headers: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(DEFAULT_HEADERS)
    if headers:
        merged.update(headers)
    return merged


def fetch_html(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> str:
    response = requests.get(
        url,
        headers=build_headers(headers),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text
