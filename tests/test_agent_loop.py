"""Agent loop: tool dispatch, feedback, termination, budgets, bad-arg self-correction."""

from __future__ import annotations

import json

from frag.agent.loop import AgentLoop
from frag.agent.tools import ToolRegistry, make_calc_tool

_SYSTEM = "test agent"


class _ScriptedLLM:
    """Returns queued replies; records the messages it saw each turn."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []
        self.last_cost = 0.001

    def chat(self, messages, tools=None):
        self.calls.append([m.get("role") for m in messages])
        return self._replies.pop(0)


def _tool_call(name, args, cid="c1"):
    return {
        "content": None,
        "tool_calls": [{"id": cid, "name": name, "arguments": json.dumps(args)}],
    }


def _answer(text):
    return {"content": text, "tool_calls": []}


def _loop(llm, **kw):
    return AgentLoop(llm, ToolRegistry([make_calc_tool()]), _SYSTEM, **kw)


def test_runs_tool_then_answers():
    llm = _ScriptedLLM(
        [_tool_call("financial_calc", {"expression": "4200 / 1000"}), _answer("4.2x")]
    )
    out = _loop(llm).run("what is leverage")
    assert out["status"] == "answered" and out["answer"] == "4.2x"
    assert out["tool_calls"] == 1
    # the tool result was fed back before the final answer
    assert "tool" in llm.calls[1]
    assert any(step["tool"] == "financial_calc" for step in out["trace"])


def test_total_cost_accumulates():
    llm = _ScriptedLLM([_tool_call("financial_calc", {"expression": "1 + 1"}), _answer("2")])
    out = _loop(llm).run("q")
    assert out["total_cost"] == 0.002  # two chat() calls at 0.001 each


def test_max_turns_budget_stops():
    # Always asks for a tool -> never answers; must stop, not loop forever.
    llm = _ScriptedLLM([_tool_call("financial_calc", {"expression": "1+1"})] * 10)
    out = _loop(llm, max_turns=3, max_tool_calls=99).run("q")
    assert out["status"] == "stopped:max_turns"


def test_max_tool_calls_budget_stops():
    llm = _ScriptedLLM([_tool_call("financial_calc", {"expression": "1+1"})] * 10)
    out = _loop(llm, max_turns=99, max_tool_calls=2).run("q")
    assert out["status"] == "stopped:max_tool_calls" and out["tool_calls"] == 2


def test_bad_tool_args_are_fed_back_not_crash():
    # First call omits the required 'expression'; the loop feeds the error back, then answers.
    llm = _ScriptedLLM([_tool_call("financial_calc", {}), _answer("recovered")])
    out = _loop(llm).run("q")
    assert out["status"] == "answered" and out["answer"] == "recovered"


def test_unknown_tool_is_reported_not_crash():
    llm = _ScriptedLLM([_tool_call("nope", {"x": 1}), _answer("ok")])
    out = _loop(llm).run("q")
    assert out["status"] == "answered"
