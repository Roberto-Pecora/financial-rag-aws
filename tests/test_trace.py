"""Timing + summary statistics."""

from __future__ import annotations

from frag.eval import trace


def test_timer_measures_positive_elapsed():
    with trace.Timer() as t:
        sum(range(1000))
    assert t.elapsed >= 0.0


def test_time_call_returns_result_and_elapsed():
    result, elapsed = trace.time_call(lambda x: x * 2, 21)
    assert result == 42 and elapsed >= 0.0


def test_percentile_interpolates():
    vals = [10, 20, 30, 40]
    assert trace.percentile(vals, 50) == 25.0
    assert trace.percentile(vals, 0) == 10
    assert trace.percentile([], 50) == 0.0
    assert trace.percentile([5], 95) == 5


def test_latency_summary_ms():
    out = trace.latency_summary([0.1, 0.2, 0.3])  # seconds
    assert out["p50_ms"] == 200.0
    assert out["p95_ms"] >= out["p50_ms"]


def test_mean_sd():
    m, sd = trace.mean_sd([2.0, 4.0])
    assert m == 3.0 and sd > 0
    assert trace.mean_sd([5.0]) == (5.0, 0.0)
    assert trace.mean_sd([]) == (0.0, 0.0)
