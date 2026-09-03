"""OpenRouter cost estimation."""

from __future__ import annotations

from frag.eval import pricing


def test_estimate_cost_known_model():
    # 1M prompt @ 0.12 + 1M completion @ 0.30 = 0.42
    cost = pricing.estimate_cost("meta-llama/llama-3.3-70b-instruct", 1_000_000, 1_000_000)
    assert round(cost, 4) == 0.42


def test_estimate_cost_unknown_model_is_zero():
    assert pricing.estimate_cost("mystery/model", 1_000_000, 1_000_000) == 0.0


def test_cost_from_usage():
    usage = {"prompt_tokens": 500_000, "completion_tokens": 100_000}
    cost = pricing.cost_from_usage("meta-llama/llama-3.3-70b-instruct", usage)
    assert round(cost, 6) == round((500_000 * 0.12 + 100_000 * 0.30) / 1_000_000, 6)


def test_cost_from_usage_none():
    assert pricing.cost_from_usage("any", None) == 0.0


def test_load_prices_from_file(tmp_path):
    import json

    p = tmp_path / "prices.json"
    p.write_text(json.dumps({"x/y": [1.0, 2.0]}))
    prices = pricing.load_prices(str(p))
    assert prices["x/y"] == (1.0, 2.0)
