"""History-aware query rewriting: resolve a follow-up into a standalone question."""

from __future__ import annotations

from typing import Any

_PROMPT = (
    "Rewrite the follow-up into a standalone question that needs no prior context. "
    "Resolve pronouns and references using the history. Reply with the question only.\n\n"
    "History:\n{history}\n\nFollow-up: {question}"
)


def rewrite_query(question: str, history: list[dict[str, str]] | None, llm: Any | None) -> str:
    """Standalone question. No history or no llm -> the question unchanged (no call)."""
    if not history or llm is None:
        return question
    convo = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history)
    rewritten = llm.generate(_PROMPT.format(history=convo, question=question)).strip()
    return rewritten or question
