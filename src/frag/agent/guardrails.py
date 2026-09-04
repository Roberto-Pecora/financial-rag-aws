"""Input/output guardrails: screen queries for injection, answers for scope/leakage.

Deterministic and injectable — no key needed. An optional LLM input classifier sits
behind INPUT_GUARD_LLM (off by default).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Direct override / jailbreak markers seen in the query itself.
_OVERRIDE_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |your )?(system|previous|prior) (prompt|instructions)",
    r"reveal (your |the )?(system prompt|instructions|prompt)",
    r"you are now",
    r"pretend to be",
    r"developer mode",
    r"jailbreak",
]
_MAX_QUERY_CHARS = 4000

# Light leakage scan on answers.
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_SECRET = re.compile(r"\b(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})\b")


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str = ""


def screen_input(text: str) -> Verdict:
    """Flag override/jailbreak attempts and absurd length in the user query."""
    if len(text) > _MAX_QUERY_CHARS:
        return Verdict(False, "query too long")
    low = text.lower()
    for pat in _OVERRIDE_PATTERNS:
        if re.search(pat, low):
            logger.warning("input guardrail blocked query: %s", pat)
            return Verdict(False, "possible prompt-injection / override attempt")
    return Verdict(True)


def _norm_label(s: str) -> str:
    """Normalise a citation/label so formatting variance (a 'doc-' prefix, brackets,
    case) doesn't read as a fabricated citation."""
    s = s.strip().strip("[]").lower()
    for prefix in ("doc-", "doc ", "document "):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s.strip()


def screen_output(answer: str, citations: list[str], allowed_labels: set[str]) -> Verdict:
    """Reject fabricated citations and flag leaked PII/secrets in the answer.

    A citation counts as valid if, after normalisation, it matches (or is a prefix
    of) any retrieved label — so a real answer isn't withheld over 'doc-'/case quirks.
    """
    norm_allowed = [_norm_label(a) for a in allowed_labels]

    def cited_ok(c: str) -> bool:
        nc = _norm_label(c)
        if not nc:
            return False
        return any(nc == a or a.startswith(nc) or nc.startswith(a) for a in norm_allowed)

    invalid = [c for c in citations if not cited_ok(c)]
    if invalid:
        logger.warning("output guardrail: citations outside retrieved set: %s", invalid)
        return Verdict(False, "answer cites documents that were not retrieved")
    if _EMAIL.search(answer) or _SECRET.search(answer):
        logger.warning("output guardrail: answer contains PII/secret-like text")
        return Verdict(False, "answer contains PII or secret-like content")
    return Verdict(True)
