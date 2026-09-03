"""End-to-end bench aggregation with a fake controller (no network)."""

from __future__ import annotations

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "bench_end_to_end", pathlib.Path(__file__).parent.parent / "scripts" / "bench_end_to_end.py"
)
bench_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_mod)


class _FakeClient:
    last_cost = 0.002


class _FakeActor:
    client = _FakeClient()


class _FakeController:
    actor = _FakeActor()

    def answer_with_critique(self, query, top_k=8):
        return {"query": query, "answer": "x"}


def test_bench_reports_latency_and_cost():
    golden = [{"query": "a"}, {"query": "b"}]
    out = bench_mod.bench(_FakeController(), golden, repeats=2)
    assert "p50_ms" in out and "p95_ms" in out
    assert round(out["cost_mean_usd"], 3) == 0.002
    assert out["cost_sd_usd"] == 0.0  # constant cost -> zero sd
