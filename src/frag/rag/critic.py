from __future__ import annotations

import json
from typing import Protocol

from frag.rag import prompts
from frag.rag.openrouter_client import OpenRouterClient


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

        prompt = self.build_prompt(query, contexts, answer, citations)
        text = self.client.generate(prompt)

        try:
            data = json.loads(text.strip())
            overall = float(data.get("overall_score", 0.0))
            faithfulness = float(data.get("faithfulness_score", 0.0))
            completeness = float(data.get("completeness_score", 0.0))
            citation_score = float(data.get("citation_score", 0.0))
            issues = data.get("issues", [])
            if not isinstance(issues, list):
                issues = [str(issues)]

            notes_parts = [
                f"faithfulness={faithfulness:.2f}",
                f"completeness={completeness:.2f}",
                f"citations={citation_score:.2f}",
            ]
            if issues:
                notes_parts.append("issues: " + " | ".join(str(i) for i in issues))

            return {
                "score": overall,
                "notes": "; ".join(notes_parts),
            }
        except Exception:
            return {"score": 0.0, "notes": "Critic failed to parse JSON."}
