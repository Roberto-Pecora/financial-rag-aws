"""LLM-boundary Pydantic models: coercion, defaults, and rejection of bad shapes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from frag.rag.llm_schemas import ActorResponse, CriticResponse


def test_actor_response_coerces_and_defaults():
    r = ActorResponse.model_validate_json('{"answer": "  hi  ", "citations": ["d1", 2]}')
    assert r.answer == "hi"  # stripped
    assert r.citations == ["d1", "2"]  # each coerced to str


def test_actor_response_bad_citations_become_empty():
    r = ActorResponse.model_validate_json('{"answer": "x", "citations": "d1"}')
    assert r.citations == []  # non-list -> []


def test_actor_response_missing_fields_default():
    r = ActorResponse.model_validate_json("{}")
    assert r.answer == "" and r.citations == []


def test_actor_response_rejects_non_object():
    with pytest.raises(ValidationError):
        ActorResponse.model_validate_json("[1, 2, 3]")


def test_critic_response_coerces_numeric_strings():
    r = CriticResponse.model_validate_json('{"overall_score": "0.9"}')
    assert r.overall_score == 0.9


def test_critic_response_rejects_non_numeric_score():
    with pytest.raises(ValidationError):
        CriticResponse.model_validate_json('{"overall_score": "high"}')


def test_critic_response_issues_coercion():
    r = CriticResponse.model_validate_json('{"issues": "one problem"}')
    assert r.issues == ["one problem"]  # scalar wrapped into a list
