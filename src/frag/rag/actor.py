from __future__ import annotations

import json
from typing import Protocol

from frag.rag import prompts
from frag.rag.openrouter_client import OpenRouterClient


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


def _doc_id(ctx: dict, idx: int) -> str:
    """Resolve a stable identifier from a context dict returned by QdrantStore.search().

    The store returns {text, metadata, score}. The original doc id lives inside
    metadata under 'doc_id', 'id', or 'ticker' — fall back to a positional label.
    """
    meta = ctx.get("metadata") or {}
    return meta.get("doc_id") or meta.get("id") or meta.get("ticker") or f"doc-{idx + 1}"


class Actor:
    def __init__(self, llm_client: LLMClient | None = None, prompt_version: int | None = None):
        self.client = llm_client or OpenRouterClient("ACTOR_MODEL")
        version = prompt_version or prompts.resolve_version("actor", "ACTOR_PROMPT_VERSION")
        self.prompt = prompts.get("actor", version)
        self.prompt_version = self.prompt.version

    def build_prompt(self, query: str, contexts: list) -> str:
        evidence = "\n\n".join([f"[{_doc_id(c, i)}] {c['text']}" for i, c in enumerate(contexts)])
        return self.prompt.render(query=query, evidence=evidence)

    def act(self, query: str, contexts: list) -> dict:
        if not contexts:
            raw = '{"answer":"Insufficient evidence retrieved.","citations":[]}'
            return {"answer": "Insufficient evidence retrieved.", "citations": [], "raw": raw}

        prompt = self.build_prompt(query, contexts)
        raw = self.client.generate(prompt)

        try:
            data = json.loads(raw)
            answer = str(data.get("answer", "")).strip()
            citations = data.get("citations", [])
            if not isinstance(citations, list):
                citations = []
            citations = [str(c).strip() for c in citations]
            return {"answer": answer, "citations": citations, "raw": raw}
        except Exception:
            return {
                "answer": "Insufficient evidence retrieved.",
                "citations": [],
                "raw": raw,
            }
