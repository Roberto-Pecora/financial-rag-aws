"""Top-level entry: rewrite a follow-up, route it, and optionally critic-gate the agent.

lookup -> deterministic RagController (its own actor/critic + guardrails);
multi_hop -> the tool-loop, whose final answer the critic can veto as a risk gate.
"""

from __future__ import annotations

from typing import Any

from frag.agent import router
from frag.agent.rewrite import rewrite_query


class Orchestrator:
    def __init__(
        self,
        controller: Any,
        agent: Any,
        rewrite_llm: Any | None = None,
        route_llm: Any | None = None,
        critic: Any | None = None,
        min_score: float = 0.8,
    ) -> None:
        self.controller = controller
        self.agent = agent
        self.rewrite_llm = rewrite_llm
        self.route_llm = route_llm
        self.critic = critic
        self.min_score = min_score

    def answer(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        standalone = rewrite_query(question, history, self.rewrite_llm)
        route = router.route(standalone, self.route_llm)

        if route == "lookup":
            payload = self.controller.answer_with_critique(standalone)
            return {"route": "lookup", "question": standalone, **payload}

        result = self.agent.run(standalone)
        result = self._gate(standalone, result)
        return {"route": "multi_hop", "question": standalone, **result}

    def _gate(self, question: str, result: dict[str, Any]) -> dict[str, Any]:
        """Critic vetoes an agent answer that the gathered evidence does not support."""
        if self.critic is None or result.get("status") != "answered":
            return result
        critic_out = self.critic.critique(
            question, result.get("evidence", []), result["answer"], []
        )
        result["critic_score"] = critic_out["score"]
        result["critic_notes"] = critic_out["notes"]
        if critic_out["score"] < self.min_score:
            result["status"] = "abstained"
            result["answer"] = "Answer withheld — critic could not verify it against the evidence."
        return result
