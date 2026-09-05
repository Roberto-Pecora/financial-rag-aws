"""Corrective retrieval (CRAG-style): grade passages against the query, and if too
few are relevant, rewrite the query and re-retrieve. Off by default (CORRECTIVE)."""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel, ValidationError

from frag.rag import prompts
from frag.rag.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


def corrective_enabled() -> bool:
    return os.getenv("CORRECTIVE", "off").strip().lower() in {"on", "1", "true", "yes"}


class DocGrade(BaseModel):
    relevant: bool = False


class RewriteOut(BaseModel):
    query: str = ""


class RetrievalGrader:
    """Keep only passages an LLM judges relevant to the query. Fails open."""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm or OpenRouterClient("GRADER_MODEL", response_model=DocGrade)

    def keep_relevant(self, query: str, contexts: list[dict]) -> list[dict]:
        kept = []
        for c in contexts:
            raw = self.llm.generate(
                prompts.get("grader").render(query=query, passage=c.get("text", "")[:2000])
            )
            try:
                if DocGrade.model_validate_json(raw).relevant:
                    kept.append(c)
            except ValidationError:
                logger.warning("grader response malformed; keeping passage")
                kept.append(c)  # don't drop evidence on a parse error
        return kept


class RetrievalRewriter:
    """Reformulate a query for better recall (distinct from multi-turn rewriting)."""

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm or OpenRouterClient("GRADER_MODEL", response_model=RewriteOut)

    def rewrite(self, query: str) -> str:
        raw = self.llm.generate(prompts.get("retrieval_rewrite").render(query=query))
        try:
            return RewriteOut.model_validate_json(raw).query.strip() or query
        except ValidationError:
            return query
