"""Timing and summary stats for latency/cost eval: Timer, percentiles, mean+-sd."""

from __future__ import annotations

import time
from statistics import fmean, stdev
from typing import Any


class Timer:
    """Context manager exposing elapsed wall-clock seconds as `.elapsed`."""

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


def time_call(fn, *args, **kwargs):
    """Run `fn`, returning (result, elapsed_seconds)."""
    with Timer() as t:
        result = fn(*args, **kwargs)
    return result, t.elapsed


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile, p in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def latency_summary(seconds: list[float]) -> dict[str, float]:
    """p50/p95 in milliseconds from per-call seconds."""
    ms = [s * 1000.0 for s in seconds]
    return {"p50_ms": percentile(ms, 50), "p95_ms": percentile(ms, 95)}


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample sd (sd=0 for a single value)."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return fmean(values), stdev(values)
