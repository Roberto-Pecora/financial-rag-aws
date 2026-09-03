from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from frag.utils.config import configure_runtime

configure_runtime()

logger = logging.getLogger(__name__)

SEC_HEADERS: dict[str, str] = {
    "User-Agent": os.getenv(
        "SEC_USER_AGENT",
        "actor-critic-financial-rag/0.1 (local research; contact: local@localhost)",
    ),
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}

RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}
RETRY_BASE_DELAY_SECONDS = 5.0
MAX_RETRIES = 6


def build_sec_headers(headers: dict[str, str] | None = None) -> dict[str, str]:
    merged = dict(SEC_HEADERS)
    if headers:
        for key, value in headers.items():
            if key.lower() == "user-agent" and value:
                merged["User-Agent"] = value
    return {
        "User-Agent": merged["User-Agent"],
        "Accept": merged["Accept"],
        "Accept-Encoding": merged["Accept-Encoding"],
    }


def _retry_delay_for_response(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
    return RETRY_BASE_DELAY_SECONDS * attempt


def request_with_retry(
    url: str, headers: dict[str, str] | None = None, timeout: int = 30
) -> requests.Response:
    last_error: Exception | None = None
    last_status: int | None = None
    last_body = ""
    request_headers = build_sec_headers(headers)

    logger.debug(
        "SEC request headers: %s",
        {
            "User-Agent": request_headers.get("User-Agent"),
            "Accept": request_headers.get("Accept"),
            "Accept-Encoding": request_headers.get("Accept-Encoding"),
        },
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = requests.Session()
            session.trust_env = False
            session.proxies = {"http": None, "https": None}

            response = session.get(
                url, headers=request_headers, timeout=timeout, allow_redirects=True
            )

            logger.debug(
                "SEC response: status=%s content-type=%s final_url=%s body_preview=%s",
                response.status_code,
                response.headers.get("content-type"),
                response.url,
                response.text[:2000],
            )

            if response.status_code == 200:
                return response

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                delay = _retry_delay_for_response(response, attempt)
                logger.warning(
                    "SEC returned %s for %s; retrying in %.1fs (attempt %d/%d)",
                    response.status_code,
                    url,
                    delay,
                    attempt,
                    MAX_RETRIES,
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            return response
        except requests.RequestException as e:
            last_error = e
            last_status = getattr(getattr(e, "response", None), "status_code", None)
            if getattr(e, "response", None) is not None:
                try:
                    last_body = e.response.text[:500]
                except Exception:
                    last_body = "<unavailable>"

            if last_status in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                delay = _retry_delay_for_response(getattr(e, "response", None), attempt)
                logger.warning(
                    "SEC request failed with %s for %s; retrying in %.1fs (attempt %d/%d)",
                    last_status,
                    url,
                    delay,
                    attempt,
                    MAX_RETRIES,
                )
                time.sleep(delay)
                continue

            logger.error(
                "SEC request failed for url=%s, status=%s, body=%r",
                url,
                last_status,
                last_body,
            )
            raise

    raise RuntimeError(
        f"SEC request failed for url={url}, last_status={last_status}, body={last_body!r}"
    ) from last_error


def fetch_company_tickers_json(
    headers: dict[str, str] | None = None, timeout: int = 30
) -> dict[str, Any]:
    url = "https://www.sec.gov/files/company_tickers.json"
    response = request_with_retry(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_submissions_json(
    cik: str, headers: dict[str, str] | None = None, timeout: int = 30
) -> dict[str, Any]:
    cik10 = zero_pad_cik(cik)
    urls = [
        f"https://data.sec.gov/submissions/CIK{cik10}.json",
        f"https://www.sec.gov/submissions/CIK{cik10}.json",
    ]

    last_error: Exception | None = None
    last_status: int | None = None
    last_body = ""

    for url in urls:
        logger.debug("Fetching SEC submissions: %s", url)
        try:
            response = request_with_retry(url, headers=headers, timeout=timeout)
            if response.status_code != 200:
                response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            last_error = e
            last_status = getattr(getattr(e, "response", None), "status_code", None)
            if getattr(e, "response", None) is not None:
                try:
                    last_body = e.response.text[:500]
                except Exception:
                    last_body = "<unavailable>"

            logger.warning(
                "SEC submissions request failed for cik=%s, url=%s, status=%s, body=%r",
                cik10,
                url,
                last_status,
                last_body,
            )

    raise RuntimeError(
        "SEC submissions request failed "
        f"for cik={cik10}, last_status={last_status}, body={last_body!r}"
    ) from last_error


def extract_cik_digits(cik: str) -> str:
    digits = "".join(ch for ch in str(cik) if ch.isdigit())
    if not digits:
        raise ValueError(f"Invalid CIK: {cik!r}")
    return digits


def zero_pad_cik(cik: str) -> str:
    return extract_cik_digits(cik).zfill(10)


def archive_cik(cik: str) -> str:
    return str(int(extract_cik_digits(cik)))
