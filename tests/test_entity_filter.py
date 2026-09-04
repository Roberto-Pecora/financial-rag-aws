"""Entity-aware retrieval: company matching, aliases, and the keyword-field filter."""

from __future__ import annotations

from frag.rag import entity_filter as ef
from frag.rag import store_opensearch as sos

_COMPANIES = ["Amazon", "Coca-Cola", "PepsiCo", "JPMorgan", "American Express", "Best Buy"]


def test_matches_named_company():
    assert ef.match_company("What was Amazon's operating cash flow?", _COMPANIES) == "Amazon"


def test_no_match_returns_none():
    assert ef.match_company("What was total revenue?", _COMPANIES) is None


def test_multiword_company():
    assert ef.match_company("Best Buy revenue trend", _COMPANIES) == "Best Buy"


def test_alias_resolves_to_canonical():
    assert ef.match_company("how did Coke do", _COMPANIES) == "Coca-Cola"


def test_company_filter_shape():
    assert ef.company_filter("Amazon cash flow", _COMPANIES) == {"company": "Amazon"}
    assert ef.company_filter("generic question", _COMPANIES) is None


def test_store_filter_targets_keyword_subfield():
    """A company filter must term-match company.keyword, not the analysed text field."""
    q = sos.knn_query([0.1], top_k=5, metadata_filter={"company": "Amazon"})
    must = q["query"]["bool"]["must"]
    assert {"term": {"company.keyword": "Amazon"}} in must


def test_store_filter_leaves_keyword_fields_untouched():
    q = sos.knn_query([0.1], top_k=5, metadata_filter={"ticker": "AAPL"})
    must = q["query"]["bool"]["must"]
    assert {"term": {"ticker": "AAPL"}} in must  # already keyword, no .keyword appended
