"""Corrective retrieval: grading, rewrite parsing, and the controller retry loop."""

from __future__ import annotations

import json

from frag.rag.controller import RagController
from frag.rag.grader import RetrievalGrader, RetrievalRewriter

_CTX = [
    {"text": "Amazon net sales were 5,678.", "metadata": {"doc_id": "d1"}},
    {"text": "Unrelated risk factors.", "metadata": {"doc_id": "d2"}},
]


class _GradeLLM:
    """Returns relevant=True only for passages containing 'net sales'."""

    def __init__(self):
        self.seen = []

    def generate(self, prompt):
        self.seen.append(prompt)
        return json.dumps({"relevant": "net sales" in prompt})


class _RewriteLLM:
    def generate(self, prompt):
        return json.dumps({"query": "Amazon cash provided by operating activities"})


# -- grader / rewriter units -----------------------------------------------


def test_grader_keeps_only_relevant():
    kept = RetrievalGrader(llm=_GradeLLM()).keep_relevant("q", _CTX)
    assert [c["metadata"]["doc_id"] for c in kept] == ["d1"]


def test_grader_fails_open_on_bad_json():
    class _Bad:
        def generate(self, prompt):
            return "not json"

    kept = RetrievalGrader(llm=_Bad()).keep_relevant("q", _CTX)
    assert len(kept) == 2  # nothing dropped on a parse error


def test_rewriter_parses_query():
    assert RetrievalRewriter(llm=_RewriteLLM()).rewrite("Amazon operating cash flow") == (
        "Amazon cash provided by operating activities"
    )


def test_rewriter_falls_back_on_bad_json():
    class _Bad:
        def generate(self, prompt):
            return "{bad"

    assert RetrievalRewriter(llm=_Bad()).rewrite("q") == "q"


# -- controller corrective loop --------------------------------------------


class _Store:
    """Returns weak docs on the original query, strong docs after a rewrite."""

    def __init__(self):
        self.queries = []

    def search(self, query, top_k=8, metadata_filter=None):
        self.queries.append(query)
        if "operating activities" in query:  # the rewritten query
            return [{"text": "Amazon net sales and cash flow.", "metadata": {"doc_id": "good"}}]
        return [{"text": "Irrelevant boilerplate.", "metadata": {"doc_id": "weak"}}]

    def count(self):
        return 1


def _controller(monkeypatch, store):
    monkeypatch.setenv("CORRECTIVE", "on")
    c = RagController(store=store)
    c.grader = RetrievalGrader(llm=_GradeLLM())
    c.rewriter = RetrievalRewriter(llm=_RewriteLLM())
    c.max_rewrites = 2
    c.min_relevant = 1
    return c


def test_corrective_retry_recovers_relevant_docs(monkeypatch):
    store = _Store()
    c = _controller(monkeypatch, store)
    out = c._corrective_retrieve("Amazon operating cash flow", top_k=5, metadata_filter=None)
    assert [d["metadata"]["doc_id"] for d in out] == ["good"]  # rewrite rescued it
    assert len(store.queries) >= 2  # original + at least one rewrite


def test_corrective_off_does_single_search(monkeypatch):
    monkeypatch.delenv("CORRECTIVE", raising=False)
    store = _Store()
    c = RagController(store=store)  # corrective off -> grader is None
    assert c.grader is None
    c._corrective_retrieve("q", top_k=5, metadata_filter=None)
    assert len(store.queries) == 1  # no grading, no rewrite
