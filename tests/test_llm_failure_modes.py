"""Actor/Critic failure handling: transport errors surface, bad responses degrade."""

from __future__ import annotations

import json

import pytest

from frag.rag.actor import Actor
from frag.rag.critic import Critic

_CTX = [{"text": "Revenue was 5,678.", "metadata": {"doc_id": "d1"}}]


class _RaisingLLM:
    def generate(self, prompt):
        raise ConnectionError("network down")


class _RawLLM:
    def __init__(self, raw):
        self.raw = raw

    def generate(self, prompt):
        return self.raw


# -- transport/auth errors must propagate, not become abstain/veto ----------


def test_actor_propagates_transport_error():
    with pytest.raises(ConnectionError):
        Actor(llm_client=_RaisingLLM()).act("q", _CTX)


def test_critic_propagates_transport_error():
    with pytest.raises(ConnectionError):
        Critic(llm_client=_RaisingLLM()).critique("q", _CTX, "ans", ["d1"])


# -- malformed model responses degrade gracefully (and are logged) ----------


def test_actor_abstains_on_non_json(caplog):
    out = Actor(llm_client=_RawLLM("not json at all")).act("q", _CTX)
    assert out["answer"] == "Insufficient evidence retrieved." and out["citations"] == []
    assert "malformed" in caplog.text


def test_actor_abstains_on_non_object_json():
    out = Actor(llm_client=_RawLLM("[1, 2, 3]")).act("q", _CTX)
    assert out["answer"] == "Insufficient evidence retrieved."


def test_critic_vetoes_on_non_json(caplog):
    out = Critic(llm_client=_RawLLM("garbage")).critique("q", _CTX, "ans", ["d1"])
    assert out["score"] == 0.0 and "malformed" in out["notes"].lower()
    assert "malformed" in caplog.text


def test_critic_vetoes_on_non_numeric_score():
    """Valid JSON but a non-numeric score is malformed, not a real 0.0 verdict."""
    raw = json.dumps({"overall_score": "high"})
    out = Critic(llm_client=_RawLLM(raw)).critique("q", _CTX, "ans", ["d1"])
    assert out["score"] == 0.0 and "malformed" in out["notes"].lower()


def test_actor_parses_well_formed_response():
    raw = json.dumps({"answer": "Revenue was 5,678.", "citations": ["d1"]})
    out = Actor(llm_client=_RawLLM(raw)).act("q", _CTX)
    assert out["answer"] == "Revenue was 5,678." and out["citations"] == ["d1"]
