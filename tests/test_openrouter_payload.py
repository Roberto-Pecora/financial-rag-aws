"""OpenRouterClient payload construction (pure, no network)."""

from __future__ import annotations

from frag.rag.llm_schemas import ActorResponse
from frag.rag.openrouter_client import OpenRouterClient


def test_default_requests_json_object():
    client = OpenRouterClient("ACTOR_MODEL", response_model=ActorResponse)
    payload = client._build_payload("hello")
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["content"] == "hello"
    assert payload["temperature"] == 0.0


def test_structured_output_uses_schema_when_enabled(monkeypatch):
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT", "on")
    client = OpenRouterClient("ACTOR_MODEL", response_model=ActorResponse)
    rf = client._build_payload("hi")["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "actorresponse"
    assert "properties" in rf["json_schema"]["schema"]  # a real JSON schema


def test_structured_output_ignored_without_model(monkeypatch):
    """No response_model -> plain json_object even if the flag is on."""
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT", "on")
    client = OpenRouterClient("ACTOR_MODEL")
    assert client._build_payload("hi")["response_format"] == {"type": "json_object"}
