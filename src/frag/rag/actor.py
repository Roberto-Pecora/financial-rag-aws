from __future__ import annotations

import logging
from typing import Protocol

from pydantic import ValidationError

from frag.rag import prompts
from frag.rag.llm_schemas import ActorResponse
from frag.rag.openrouter_client import OpenRouterClient
from frag.rag.untrusted import render_evidence

logger = logging.getLogger(__name__)

_ABSTAIN = "Insufficient evidence retrieved."


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
        self.client = llm_client or OpenRouterClient("ACTOR_MODEL", response_model=ActorResponse)
        version = prompt_version or prompts.resolve_version("actor", "ACTOR_PROMPT_VERSION")
        self.prompt = prompts.get("actor", version)
        self.prompt_version = self.prompt.version

    def build_prompt(self, query: str, contexts: list) -> str:
        evidence = render_evidence(contexts, _doc_id)
        return self.prompt.render(query=query, evidence=evidence)

    def act(self, query: str, contexts: list) -> dict:
        if not contexts:
            raw = f'{{"answer":"{_ABSTAIN}","citations":[]}}'
            return {"answer": _ABSTAIN, "citations": [], "raw": raw}

        # A transport/auth error here propagates deliberately: a failed call must
        # not masquerade as a genuine "insufficient evidence" abstention.
        prompt = self.build_prompt(query, contexts)
        raw = self.client.generate(prompt)

        # A malformed response is a logged abstention, not a silent one.
        try:
            resp = ActorResponse.model_validate_json(raw)
        except ValidationError as exc:
            logger.warning("actor response malformed (%s); abstaining. raw=%.200r", exc, raw)
            return {"answer": _ABSTAIN, "citations": [], "raw": raw}

        return {"answer": resp.answer, "citations": resp.citations, "raw": raw}
