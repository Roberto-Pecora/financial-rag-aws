"""Entity-aware retrieval: pull a known company from the query and filter to it.

Without this, "Amazon's operating cash flow" competes every company's cash-flow
passage on text similarity alone, so another company's chunk can outrank Amazon's.
Constraining retrieval to the named company fixes that. The company list is
supplied by the store (its indexed values), so this stays data-driven and testable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# A few common aliases -> canonical company name (as indexed).
_ALIASES = {
    "coke": "Coca-Cola",
    "jp morgan": "JPMorgan",
    "j&j": "Johnson & Johnson",
    "amex": "American Express",
}


def match_company(query: str, companies: Iterable[str]) -> str | None:
    """Return the known company named in the query (longest match), else None."""
    low = query.lower()
    hits = [c for c in companies if re.search(rf"\b{re.escape(c.lower())}\b", low)]
    for alias, canonical in _ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", low):
            hits.append(canonical)
    return max(hits, key=len) if hits else None


def company_filter(query: str, companies: Iterable[str]) -> dict[str, str] | None:
    """A metadata filter constraining retrieval to the query's company, or None."""
    company = match_company(query, companies)
    return {"company": company} if company else None
