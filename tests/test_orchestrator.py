"""Orchestrator: rewrite -> route -> RAG/agent, plus the critic gate on agent answers."""

from __future__ import annotations

from frag.agent.orchestrator import Orchestrator
from frag.agent.rewrite import rewrite_query


class _Controller:
    def answer_with_critique(self, query, **kw):
        return {"status": "accepted", "answer": f"lookup:{query}", "citations": []}


class _Agent:
    def __init__(self, result):
        self.result = result
        self.seen = None

    def run(self, question):
        self.seen = question
        return dict(self.result)


class _Critic:
    def __init__(self, score):
        self.score = score

    def critique(self, q, evidence, answer, citations):
        return {"score": self.score, "notes": f"score={self.score}"}


class _LLM:
    def __init__(self, raw):
        self.raw = raw

    def generate(self, prompt):
        return self.raw


# -- rewrite ---------------------------------------------------------------


def test_rewrite_noop_without_history():
    assert rewrite_query("and the margin?", None, _LLM("x")) == "and the margin?"


def test_rewrite_uses_history():
    out = rewrite_query(
        "and its margin?",
        [{"role": "user", "content": "what was Acme revenue"}],
        _LLM("What was Acme's margin?"),
    )
    assert out == "What was Acme's margin?"


# -- routing ---------------------------------------------------------------


def test_lookup_routes_to_controller():
    orch = Orchestrator(_Controller(), _Agent({"status": "answered"}))
    out = orch.answer("what was total revenue?")
    assert out["route"] == "lookup" and out["answer"].startswith("lookup:")


def test_multi_hop_routes_to_agent():
    agent = _Agent({"status": "answered", "answer": "4.2x", "evidence": []})
    out = Orchestrator(_Controller(), agent).answer("compare Acme and Beta leverage")
    assert out["route"] == "multi_hop" and out["answer"] == "4.2x"


# -- critic gate on the agent ----------------------------------------------


def test_critic_gate_accepts_grounded_answer():
    agent = _Agent({"status": "answered", "answer": "4.2x", "evidence": [{"text": "e"}]})
    orch = Orchestrator(_Controller(), agent, critic=_Critic(0.9), min_score=0.8)
    out = orch.answer("which have leverage above 4x and a covenant")
    assert out["status"] == "answered" and out["critic_score"] == 0.9


def test_critic_gate_vetoes_unverified_answer():
    agent = _Agent({"status": "answered", "answer": "made up", "evidence": []})
    orch = Orchestrator(_Controller(), agent, critic=_Critic(0.3), min_score=0.8)
    out = orch.answer("which have leverage above 4x and a covenant")
    assert out["status"] == "abstained" and "withheld" in out["answer"].lower()
