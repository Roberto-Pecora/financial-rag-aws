"""Training-pair mining: negatives exclude the positive, generation parses, BM25 ranks."""

from __future__ import annotations

import json

from frag.train import pair_mining as pm


class _FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k):
        return self._hits[:top_k]


class _FakeGenerator:
    def generate_queries(self, passage, n):
        return [f"q about {passage[:5]} #{i}" for i in range(n)]


class _FakeLLM:
    def __init__(self, raw):
        self._raw = raw

    def generate(self, prompt):
        return self._raw


def test_mine_hard_negatives_excludes_positive_and_dedupes():
    hits = [
        {"id": "p", "text": "positive"},
        {"id": "n1", "text": "neg one"},
        {"id": "n1", "text": "neg one dup"},
        {"id": "n2", "text": "neg two"},
        {"id": "n3", "text": "neg three"},
    ]
    negs = pm.mine_hard_negatives("q", "p", _FakeRetriever(hits), n_negatives=2)
    ids = [n["id"] for n in negs]
    assert ids == ["n1", "n2"]  # positive skipped, dup collapsed, capped at 2
    assert all("text" in n for n in negs)


def test_build_training_pairs_shape():
    corpus = [{"id": "d1", "text": "Revenue rose to 100 in 2026."}]
    retr = _FakeRetriever([{"id": "d2", "text": "unrelated"}, {"id": "d3", "text": "other"}])
    pairs = pm.build_training_pairs(
        corpus, _FakeGenerator(), retr, queries_per_passage=2, n_negatives=1
    )
    assert len(pairs) == 2
    assert pairs[0]["positive_id"] == "d1"
    assert pairs[0]["positive"].startswith("Revenue")
    assert len(pairs[0]["negatives"]) == 1


def test_synthetic_query_generator_parses_json():
    gen = pm.SyntheticQueryGenerator(
        llm=_FakeLLM('{"queries": ["What was revenue?", "Margin trend?"]}')
    )
    qs = gen.generate_queries("passage text", n=2)
    assert qs == ["What was revenue?", "Margin trend?"]


def test_synthetic_query_generator_handles_bad_json():
    gen = pm.SyntheticQueryGenerator(llm=_FakeLLM("not json"))
    assert gen.generate_queries("x", n=2) == []


def test_synthetic_query_generator_propagates_transport_error():
    import pytest

    class _RaisingLLM:
        def generate(self, prompt):
            raise ConnectionError("network down")

    with pytest.raises(ConnectionError):
        pm.SyntheticQueryGenerator(llm=_RaisingLLM()).generate_queries("x", n=2)


def test_synthetic_query_generator_caps_n():
    gen = pm.SyntheticQueryGenerator(llm=_FakeLLM('{"queries": ["a", "b", "c"]}'))
    assert gen.generate_queries("x", n=2) == ["a", "b"]


def test_inmemory_retriever_ranks_lexical_overlap():
    corpus = [
        {"id": "rev", "text": "Total revenue increased to 5,000 this year."},
        {"id": "risk", "text": "Supply chain risks and regulatory pressure remain."},
        {"id": "cash", "text": "Cash flow from operations was strong."},
    ]
    retr = pm.InMemoryRetriever(corpus)
    top = retr.search("what was total revenue", top_k=1)
    assert top[0]["id"] == "rev"


def test_write_pairs_jsonl(tmp_path):
    pairs = [{"query": "q", "positive": "p", "positive_id": "d1", "negatives": []}]
    out = str(tmp_path / "pairs.jsonl")
    n = pm.write_pairs_jsonl(pairs, out)
    assert n == 1
    assert json.loads(open(out, encoding="utf-8").readline())["query"] == "q"
