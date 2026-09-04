from __future__ import annotations

import logging
from typing import Protocol

from pydantic import ValidationError

from frag.rag import prompts
from frag.rag.llm_schemas import CriticResponse
from frag.rag.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


def _doc_id(ctx: dict, idx: int) -> str:
    """Resolve a stable identifier from a context dict returned by QdrantStore.search()."""
    meta = ctx.get("metadata") or {}
    return meta.get("doc_id") or meta.get("id") or meta.get("ticker") or f"doc-{idx + 1}"


class Critic:
    def __init__(self, llm_client: LLMClient | None = None, prompt_version: int | None = None):
        self.client = llm_client or OpenRouterClient("CRITIC_MODEL")
        version = prompt_version or prompts.resolve_version("critic", "CRITIC_PROMPT_VERSION")
        self.prompt = prompts.get("critic", version)
        self.prompt_version = self.prompt.version

    def build_prompt(self, query: str, contexts: list, answer: str, citations: list) -> str:
        evidence = "\n\n".join([f"[{_doc_id(c, i)}] {c['text']}" for i, c in enumerate(contexts)])
        citations_str = ", ".join(citations) if citations else "none"
        return self.prompt.render(
            query=query, evidence=evidence, answer=answer, citations=citations_str
        )

    def critique(self, query: str, contexts: list, answer: str, citations: list) -> dict:
        if not contexts:
            return {"score": 0.0, "notes": "No evidence available."}

        # A transport/auth error here propagates: a failed critic call must not be
        # confused with a genuine low score. A malformed *response* fails closed
        # (score 0.0 -> veto), which is the safe default for a risk gate, but is
        # logged so it is visible rather than silent.
        prompt = self.build_prompt(query, contexts, answer, citations)
        text = self.client.generate(prompt)

        try:
            resp = CriticResponse.model_validate_json(text.strip())
        except ValidationError as exc:
            logger.warning("critic response malformed (%s); vetoing. raw=%.200r", exc, text)
            return {"score": 0.0, "notes": "Critic response malformed; vetoed."}

        notes_parts = [
            f"faithfulness={resp.faithfulness_score:.2f}",
            f"completeness={resp.completeness_score:.2f}",
            f"citations={resp.citation_score:.2f}",
        ]
        if resp.issues:
            notes_parts.append("issues: " + " | ".join(resp.issues))
        return {"score": resp.overall_score, "notes": "; ".join(notes_parts)}
