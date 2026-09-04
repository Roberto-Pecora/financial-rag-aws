"""HTTP retry helper: retryable statuses back off, auth fails fast, delays honour Retry-After."""

from __future__ import annotations

import pytest
import requests

from frag.utils import http_retry


class _Resp:
    def __init__(self, status, headers=None):
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)


def _sequence(*responses):
    """A do_request that yields the given responses/exceptions in order."""
    it = iter(responses)

    def do():
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    return do


def test_retries_then_succeeds():
    slept = []
    do = _sequence(_Resp(503), _Resp(429), _Resp(200))
    r = http_retry.request_with_retry(do, base_delay=1.0, sleep=slept.append)
    assert r.status_code == 200
    assert slept == [1.0, 2.0]  # linear backoff over two retries


def test_auth_failure_is_not_retried():
    slept = []
    do = _sequence(_Resp(403), _Resp(200))
    with pytest.raises(requests.HTTPError):
        http_retry.request_with_retry(do, sleep=slept.append)
    assert slept == []  # 403 raised immediately, no retry


def test_retry_after_header_overrides_backoff():
    slept = []
    do = _sequence(_Resp(429, {"Retry-After": "7"}), _Resp(200))
    http_retry.request_with_retry(do, base_delay=1.0, sleep=slept.append)
    assert slept == [7.0]


def test_exhausts_retries_and_raises():
    slept = []
    do = _sequence(_Resp(503), _Resp(503))
    with pytest.raises(requests.HTTPError):
        http_retry.request_with_retry(do, max_retries=2, base_delay=1.0, sleep=slept.append)
    assert slept == [1.0]  # slept once, then the final attempt raised


def test_connection_error_with_retryable_response_retries():
    slept = []
    err = requests.ConnectionError("boom")
    err.response = _Resp(502)
    do = _sequence(err, _Resp(200))
    r = http_retry.request_with_retry(do, base_delay=1.0, sleep=slept.append)
    assert r.status_code == 200 and slept == [1.0]
