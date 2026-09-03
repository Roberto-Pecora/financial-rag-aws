"""Tests for content-based relevance scoring (chunk-boundary-independent)."""

from frag.eval.harness import evaluate
from frag.eval.relevance import chunk_is_relevant, extract_facts


def test_extract_facts_pulls_numbers_and_percents():
    facts = extract_facts("Operating income was $155,237 million, up 21% from $128,528 million.")
    assert "155,237" in facts
    assert "21%" in facts
    assert "128,528" in facts


def test_extract_facts_excludes_bare_years():
    assert extract_facts("results for fiscal 2026") == []


def test_chunk_relevant_ignores_whitespace():
    # Table-cell rendering splits the $ and number across lines.
    chunk = "Operating income\n$\n155,237\n$\n128,528\n21%"
    assert chunk_is_relevant(chunk, ["155,237", "21%"])


def test_chunk_not_relevant_when_missing_a_fact():
    chunk = "Operating income was 155,237."
    assert not chunk_is_relevant(chunk, ["155,237", "21%"])


def test_evaluate_scores_by_content_not_docid():
    golden = [{"query": "q", "reference_answer": "Revenue was $331,839 million, up 18%."}]
    # Gold chunk sits at rank 2 (index 1); doc_ids are irrelevant now.
    predictions = [
        {
            "query": "q",
            "retrieved_docs": [
                {"doc_id": "whatever-1", "text": "unrelated boilerplate cover page"},
                {"doc_id": "whatever-2", "text": "Revenue $331,839 up 18% this year"},
            ],
            "critic_score": 0.5,
        }
    ]
    report = evaluate(predictions, golden, k_values=(1, 3))
    row = report["rows"][0]
    assert row["recall@1"] == 0.0  # gold not at rank 1
    assert row["recall@3"] == 1.0  # gold found within top 3
    assert report["summary"]["n_scored"] == 1


def test_evaluate_skips_non_numeric_queries():
    golden = [{"query": "q", "reference_answer": "The company discussed various risks."}]
    predictions = [
        {
            "query": "q",
            "retrieved_docs": [{"doc_id": "x", "text": "risk text"}],
            "critic_score": 0.1,
        }
    ]
    report = evaluate(predictions, golden)
    # No numeric facts -> not scorable, excluded from summary counts.
    assert report["summary"]["n_scored"] == 0
    assert report["rows"][0]["scorable"] == 0
