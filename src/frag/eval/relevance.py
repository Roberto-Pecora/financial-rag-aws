"""Content-based relevance for retrieval eval.

Gold chunks are identified by whether a retrieved chunk's *text* contains the
numeric facts asserted in the golden row's reference_answer, rather than by a
pre-recorded doc_id. This makes the golden set independent of chunk
boundaries, so re-ingesting with a different chunking strategy or embedding
model needs no golden re-verification.
"""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    """Collapse whitespace so table-cell renderings like "$\\n155,237" match
    prose renderings like "$155,237" under substring search."""
    return re.sub(r"\s+", "", text)


def extract_facts(reference_answer: str) -> list[str]:
    """Pull numeric figures (normalized, $ dropped, bare years excluded) from a
    reference answer, to use as content-match keys against retrieved chunks."""
    candidates = re.findall(r"\$?\d[\d,]*(?:\.\d+)?%?", reference_answer or "")
    facts = []
    for c in candidates:
        if re.fullmatch(r"\d{4}", c):  # bare year, too common to be a useful key
            continue
        digits = re.sub(r"[^\d]", "", c)
        if len(digits) >= 3 or "." in c or "%" in c:
            facts.append(_normalize(c.lstrip("$")))
    return facts


def chunk_is_relevant(chunk_text: str, facts: list[str]) -> bool:
    """A chunk is relevant if its text contains every extracted fact."""
    if not facts:
        return False
    norm = _normalize(chunk_text)
    return all(f in norm for f in facts)
