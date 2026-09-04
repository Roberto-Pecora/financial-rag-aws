"""Versioned prompt registry: default pinning, env override, render, A/B wiring."""

from __future__ import annotations

import pytest

from frag.rag import prompts
from frag.rag.actor import Actor
from frag.rag.critic import Critic


class _FakeLLM:
    def __init__(self, raw='{"answer":"x","citations":[]}'):
        self.raw = raw
        self.seen = ""

    def generate(self, prompt):
        self.seen = prompt
        return self.raw


def test_default_is_pinned_not_latest():
    """Both roles ship >1 version, but the default stays v1 so adding v2 is opt-in."""
    assert prompts.versions("actor") == [1, 2]
    assert prompts.get("actor").version == 1
    assert prompts.get("critic").version == 1


def test_resolve_version_reads_env(monkeypatch):
    monkeypatch.setenv("ACTOR_PROMPT_VERSION", "2")
    assert prompts.resolve_version("actor", "ACTOR_PROMPT_VERSION") == 2
    monkeypatch.delenv("ACTOR_PROMPT_VERSION")
    assert prompts.resolve_version("actor", "ACTOR_PROMPT_VERSION") == 1  # falls back to default


def test_render_keeps_literal_json_braces():
    """string.Template must substitute $vars but leave the prompt's JSON braces intact."""
    out = prompts.get("actor", 1).render(query="What was revenue?", evidence="[doc-1] ...")
    assert "What was revenue?" in out
    assert '{ "answer": string' in out  # JSON schema brace survived


def test_get_unknown_version_raises():
    with pytest.raises(KeyError):
        prompts.get("actor", 99)


def test_actor_uses_selected_version():
    a1 = Actor(llm_client=_FakeLLM(), prompt_version=1)
    a2 = Actor(llm_client=_FakeLLM(), prompt_version=2)
    assert a1.prompt_version == 1 and a2.prompt_version == 2
    ctx = [{"text": "Revenue was 5,678.", "metadata": {"doc_id": "d1"}}]
    assert "investment research assistant" in a1.build_prompt("q", ctx)  # v1 wording
    assert "credit research analyst" in a2.build_prompt("q", ctx)  # v2 wording


def test_actor_version_from_env(monkeypatch):
    monkeypatch.setenv("ACTOR_PROMPT_VERSION", "2")
    assert Actor(llm_client=_FakeLLM()).prompt_version == 2


def test_critic_render_substitutes_all_fields():
    c = Critic(llm_client=_FakeLLM('{"overall_score":0.9}'), prompt_version=2)
    assert c.prompt_version == 2
    c.critique("q", [{"text": "e", "metadata": {"doc_id": "d1"}}], "ans", ["d1"])
    prompt = c.client.seen
    assert "ans" in prompt and "0.6*faithfulness" in prompt  # v2 weighting present
