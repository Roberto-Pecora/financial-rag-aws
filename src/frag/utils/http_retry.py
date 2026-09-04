"""Generic HTTP retry with backoff, honouring Retry-After.

The request is supplied as a callable so this stays transport-agnostic (GET/POST);
`sleep` is injectable so retry logic is tested without real waits. Only transient
statuses retry — auth failures (401/403) are not retried.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import requests

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def retry_after_delay(response: requests.Response | None, attempt: int, base_delay: float) -> float:
    if response is not None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                return max(float(header), 1.0)
            except ValueError:
                pass
    return base_delay * attempt


def request_with_retry(
    do_request: Callable[[], requests.Response],
    *,
    max_retries: int = 4,
    base_delay: float = 2.0,
    retryable_status: frozenset[int] = RETRYABLE_STATUS,
    sleep: Callable[[float], None] = time.sleep,
    label: str = "request",
) -> requests.Response:
    """Call `do_request`, retrying transient failures; raise the last error otherwise."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = do_request()
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in retryable_status and attempt < max_retries:
                delay = retry_after_delay(getattr(exc, "response", None), attempt, base_delay)
                logger.warning(
                    "%s failed (%s); retrying in %.1fs (%d/%d)",
                    label,
                    status,
                    delay,
                    attempt,
                    max_retries,
                )
                sleep(delay)
                continue
            raise

        if resp.status_code in retryable_status and attempt < max_retries:
            delay = retry_after_delay(resp, attempt, base_delay)
            logger.warning(
                "%s returned %s; retrying in %.1fs (%d/%d)",
                label,
                resp.status_code,
                delay,
                attempt,
                max_retries,
            )
            sleep(delay)
            continue

        resp.raise_for_status()
        return resp

    raise RuntimeError(f"{label} exhausted {max_retries} retries")
