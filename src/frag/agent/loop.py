"""Tool-loop over native function-calling, with budgets and tracing.

The model drives: it either emits tool_calls (which we validate, run, and feed
back) or a final answer. Turn and tool-call budgets bound the loop; every step is
traced with latency and cost so the run is inspectable.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from frag.agent.guardrails import screen_input
from frag.agent.tools import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoop:
    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        system_prompt: str,
        max_turns: int = 6,
        max_tool_calls: int = 8,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls

    def run(self, question: str) -> dict[str, Any]:
        verdict = screen_input(question)
        if not verdict.allowed:
            return {
                "answer": "Request blocked by the input guardrail.",
                "status": "refused",
                "trace": [],
                "evidence": [],
                "tool_calls": 0,
                "total_cost": 0.0,
            }

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        specs = self.registry.specs()
        trace: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        tool_calls_made = 0
        total_cost = 0.0

        for turn in range(1, self.max_turns + 1):
            reply = self.llm.chat(messages, tools=specs)
            total_cost += getattr(self.llm, "last_cost", 0.0)
            calls = reply.get("tool_calls") or []

            if not calls:
                trace.append({"turn": turn, "tool": None, "cost": self.llm.last_cost})
                return {
                    "answer": reply.get("content") or "",
                    "status": "answered",
                    "trace": trace,
                    "evidence": evidence,
                    "tool_calls": tool_calls_made,
                    "total_cost": total_cost,
                }

            # Capture any reasoning the model emitted alongside its tool calls.
            if reply.get("content"):
                trace.append({"turn": turn, "reasoning": reply["content"]})

            # Record the assistant turn that requested the tools, then answer each.
            messages.append(
                {"role": "assistant", "content": reply.get("content"), "tool_calls": calls}
            )
            for call in calls:
                if tool_calls_made >= self.max_tool_calls:
                    return self._budget_stop(
                        trace, evidence, tool_calls_made, total_cost, "max_tool_calls"
                    )
                tool_calls_made += 1
                result = self._run_tool(call)
                evidence.append({"text": result, "metadata": {"doc_id": f"{call.get('name')}"}})
                trace.append(
                    {
                        "turn": turn,
                        "tool": call.get("name"),
                        "args": call.get("arguments"),
                        "cost": self.llm.last_cost,
                    }
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call.get("id"), "content": result}
                )

        return self._budget_stop(trace, evidence, tool_calls_made, total_cost, "max_turns")

    def _run_tool(self, call: dict[str, Any]) -> str:
        """Validate + run a tool; a bad call becomes an error message fed back once."""
        name = call.get("name")
        tool = self.registry.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        try:
            arguments = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            return "error: tool arguments were not valid JSON"
        try:
            return tool.call(arguments)
        except ValidationError as exc:
            logger.warning("tool %s bad args: %s", name, exc)
            return f"error: invalid arguments for {name}: {exc}"

    def _budget_stop(self, trace, evidence, tool_calls_made, total_cost, reason) -> dict[str, Any]:
        logger.warning("agent stopped on %s", reason)
        return {
            "answer": "Stopped before reaching an answer (budget exhausted).",
            "status": f"stopped:{reason}",
            "trace": trace,
            "evidence": evidence,
            "tool_calls": tool_calls_made,
            "total_cost": total_cost,
        }
