"""Agent tools: safe calculator, cited retrieval/graph wrapping, schema/validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from frag.agent import tools


class _FakeStore:
    def __init__(self, hits):
        self._hits = hits
        self.seen = None

    def search(self, query, top_k=5, metadata_filter=None):
        self.seen = (query, top_k)
        return self._hits[:top_k]


_HITS = [
    {"text": "Revenue was 5,678.", "metadata": {"doc_id": "ACME_10K"}},
    {"text": "Change of Control covenant.", "metadata": {"entity": "Change of Control"}},
]


# -- safe_calc -------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected",
    [("4200 / 1000", "4.2"), ("4.2 > 4", "True"), ("-3 + 5", "2"), ("2 ** 3", "8")],
)
def test_safe_calc_evaluates(expr, expected):
    assert tools.safe_calc(expr) == expected


@pytest.mark.parametrize("expr", ["__import__('os')", "x + 1", "open('f')", "1 if True else 2"])
def test_safe_calc_rejects_unsafe(expr):
    with pytest.raises((ValueError, SyntaxError)):
        tools.safe_calc(expr)


# -- retrieval / graph wrapping --------------------------------------------


def test_retrieve_tool_cites_and_passes_args():
    store = _FakeStore(_HITS)
    tool = tools.make_retrieve_tool(store)
    out = tool.call({"query": "revenue", "top_k": 1})
    assert store.seen == ("revenue", 1)
    # hits are wrapped as untrusted, labelled by doc_id
    assert "<<<UNTRUSTED_DOC ACME_10K>>>" in out and "Revenue was 5,678." in out


def test_graph_lookup_tool_uses_entity_label():
    tool = tools.make_graph_lookup_tool(_FakeStore(_HITS))
    out = tool.call({"query": "change of control"})
    assert "<<<UNTRUSTED_DOC Change of Control>>>" in out


def test_calc_tool_runs():
    assert tools.make_calc_tool().call({"expression": "4200 / 1000"}) == "4.2"


# -- spec + validation -----------------------------------------------------


def test_tool_spec_is_function_schema():
    spec = tools.make_retrieve_tool(_FakeStore([])).spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "retrieve"
    assert "query" in spec["function"]["parameters"]["properties"]


def test_tool_call_validates_arguments():
    tool = tools.make_calc_tool()
    with pytest.raises(ValidationError):
        tool.call({})  # missing required 'expression'


def test_registry_specs_and_get():
    reg = tools.ToolRegistry([tools.make_calc_tool(), tools.make_retrieve_tool(_FakeStore([]))])
    names = {s["function"]["name"] for s in reg.specs()}
    assert names == {"financial_calc", "retrieve"}
    assert reg.get("financial_calc") is not None
    assert reg.get("nope") is None
