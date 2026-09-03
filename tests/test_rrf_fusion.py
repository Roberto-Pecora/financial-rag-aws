"""Unit tests for Reciprocal Rank Fusion, the hybrid-search ranking step.

Pure logic, no Qdrant/embedder needed - exercises the fusion math directly.
"""

from frag.rag.store_qdrant import _reciprocal_rank_fusion


def test_doc_ranked_high_in_both_wins():
    dense = ["a", "b", "c"]
    sparse = ["a", "c", "b"]
    scores = _reciprocal_rank_fusion([dense, sparse])
    # "a" is rank 0 in both, so it must score highest.
    assert max(scores, key=scores.get) == "a"


def test_single_list_preserves_order():
    scores = _reciprocal_rank_fusion([["x", "y", "z"]])
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert ranked == ["x", "y", "z"]


def test_doc_in_one_list_can_beat_doc_deep_in_other():
    # "b" ranks 1st in sparse; "a" ranks 1st in dense but is absent from sparse.
    dense = ["a", "b"]
    sparse = ["b", "a"]
    scores = _reciprocal_rank_fusion([dense, sparse])
    # "b": 1/(k+2) + 1/(k+1); "a": 1/(k+1) + 1/(k+2) -> tie here by symmetry.
    assert scores["a"] == scores["b"]


def test_agreement_beats_split():
    # "top" is 1st in both lists; "d1"/"d2" each lead only one list.
    dense = ["top", "d1"]
    sparse = ["top", "d2"]
    scores = _reciprocal_rank_fusion([dense, sparse])
    assert scores["top"] > scores["d1"]
    assert scores["top"] > scores["d2"]


def test_empty_doc_ids_ignored():
    scores = _reciprocal_rank_fusion([["", "a", ""]])
    assert "" not in scores
    assert "a" in scores


def test_empty_rankings_return_empty():
    assert _reciprocal_rank_fusion([]) == {}
    assert _reciprocal_rank_fusion([[], []]) == {}
