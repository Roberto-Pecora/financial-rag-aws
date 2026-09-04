"""Query router: heuristic classification and the optional LLM path."""

from __future__ import annotations

import pytest

from frag.agent import router


@pytest.mark.parametrize(
    "q",
    [
        "which issuers have a change-of-control covenant and leverage above 4x?",
        "compare Acme and Beta revenue",
        "which agreements have a covenant and a rating change",
        "which company has the highest leverage",
    ],
)
def test_multi_hop_questions(q):
    assert router.classify(q) == "multi_hop"


@pytest.mark.parametrize(
    "q",
    ["what was total revenue?", "who is the sponsor of Acme?", "define restricted payments"],
)
def test_lookup_questions(q):
    assert router.classify(q) == "lookup"


class _LLM:
    def __init__(self, raw):
        self.raw = raw

    def generate(self, prompt):
        return self.raw


def test_route_uses_llm_when_enabled(monkeypatch):
    monkeypatch.setenv("ROUTER_LLM", "on")
    assert router.route("anything", _LLM("multi_hop")) == "multi_hop"
    assert router.route("anything", _LLM("lookup")) == "lookup"


def test_route_ignores_llm_when_disabled(monkeypatch):
    monkeypatch.delenv("ROUTER_LLM", raising=False)
    # deterministic path wins regardless of the llm
    assert router.route("what was revenue", _LLM("multi_hop")) == "lookup"
