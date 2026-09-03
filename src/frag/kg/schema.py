"""Credit knowledge-graph schema: node and edge types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Node types (financial credit domain).
NODE_TYPES = ("Issuer", "Instrument", "Sponsor", "Covenant", "Metric", "Rating", "Event")
# Edge types linking them.
EDGE_TYPES = ("issues", "ranks_above", "owned_by", "has_covenant", "modifies", "rated_as")


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    source: str  # entity id
    type: str
    target: str  # entity id
    attrs: dict[str, Any] = field(default_factory=dict)


def entity_id(type_: str, name: str) -> str:
    """Stable id from type + normalised name, so the same entity dedupes."""
    return f"{type_.lower()}:{name.strip().lower()}"
