"""Route a question: single-fact lookup -> deterministic RAG; multi-hop -> the agent.

Deterministic by default (no key); an LLM classifier sits behind ROUTER_LLM.
"""

from __future__ import annotations

import os
import re

# Signals a question needs composition (retrieve + graph + compare), not one lookup.
_MULTIHOP_PATTERNS = [
    r"\b(and|both)\b.*\b(covenant|leverage|rating|revenue|debt)\b",
    r"\b(compare|versus|vs\.?|difference between)\b",
    r"\b(more than|greater than|less than|above|below|over|under)\s+[\d£$]",
    r"\bwhich .*\bhave\b.*\band\b",
    r"\b(highest|lowest|most|least)\b",
]


def classify(question: str) -> str:
    """Return 'multi_hop' or 'lookup'."""
    low = question.lower()
    if any(re.search(p, low) for p in _MULTIHOP_PATTERNS):
        return "multi_hop"
    return "lookup"


def route(question: str, llm=None) -> str:
    """Deterministic route, or an LLM classifier when ROUTER_LLM is on and llm given."""
    if llm is not None and os.getenv("ROUTER_LLM", "off").strip().lower() in {"on", "1", "true"}:
        raw = llm.generate(
            "Classify the question as 'lookup' (one fact) or 'multi_hop' (needs several "
            f"steps/comparison). Reply with one word only.\n\nQuestion: {question}"
        )
        return "multi_hop" if "multi" in raw.lower() else "lookup"
    return classify(question)
