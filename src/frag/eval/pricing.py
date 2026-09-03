"""OpenRouter cost estimation from token usage (USD per 1M tokens)."""

from __future__ import annotations

import json
import os

# (prompt, completion) USD per 1M tokens. Snapshot; override via PRICE_MAP_PATH.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),
    "qwen/qwen-2.5-72b-instruct": (0.12, 0.39),
    "anthropic/claude-sonnet-5": (3.00, 15.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
}


def load_prices(path: str | None = None) -> dict[str, tuple[float, float]]:
    path = path or os.getenv("PRICE_MAP_PATH")
    if not path:
        return dict(DEFAULT_PRICES)
    with open(path, encoding="utf-8") as fh:
        return {k: (float(v[0]), float(v[1])) for k, v in json.load(fh).items()}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prices: dict[str, tuple[float, float]] | None = None,
) -> float:
    """USD cost of one call; unknown models cost 0.0."""
    p_in, p_out = (prices or DEFAULT_PRICES).get(model, (0.0, 0.0))
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000.0


def cost_from_usage(model: str, usage: dict | None, prices=None) -> float:
    """Cost from an OpenRouter usage block."""
    if not usage:
        return 0.0
    return estimate_cost(
        model, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), prices
    )
