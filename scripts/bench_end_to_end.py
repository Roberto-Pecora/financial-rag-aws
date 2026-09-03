"""End-to-end query bench: total latency (p50/p95) + LLM cost (mean +- sd).

Runs the full controller pipeline (retrieve -> rerank -> actor) over the golden
queries and reads latency/cost off the instrumented OpenRouter client. Needs
OPENROUTER_API_KEY and a live/injected store.

    python scripts/bench_end_to_end.py --golden data/golden_seed.csv --repeats 3
"""

from __future__ import annotations

import argparse

import pandas as pd

from frag.eval.trace import latency_summary, mean_sd, time_call
from frag.utils.config import configure_runtime


def bench(controller, golden, repeats=1):
    """Return latency p50/p95 (ms) and cost mean+-sd over the golden queries."""
    latencies, costs = [], []
    for _ in range(repeats):
        for g in golden:
            _out, elapsed = time_call(controller.answer_with_critique, g["query"])
            latencies.append(elapsed)
            costs.append(getattr(controller.actor.client, "last_cost", 0.0))
    cost_mean, cost_sd = mean_sd(costs)
    return {**latency_summary(latencies), "cost_mean_usd": cost_mean, "cost_sd_usd": cost_sd}


def main():
    configure_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden_seed.csv")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    from frag.rag.controller import RagController

    df = pd.read_csv(args.golden)
    golden = [{"query": r["query"]} for _, r in df.iterrows()]
    print(bench(RagController(), golden, repeats=args.repeats))


if __name__ == "__main__":
    main()
