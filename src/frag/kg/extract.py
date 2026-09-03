"""Hybrid entity/relationship extraction: deterministic gazetteer + schema-guided LLM.

The gazetteer path always runs (offline, deterministic) for fixed entities
(tickers, sponsors); the LLM path (injectable, gated on the key) extracts
covenants and relationships from prose. Merge logic is pure and tested with a
fake LLM. Optional GLiNER local NER is a lazy add, not required.
"""

from __future__ import annotations

import json
import re
from typing import Any

from frag.kg.schema import Entity, Relation, entity_id


class GazetteerExtractor:
    """Match known entities by name/alias from a {type: {name: aliases}} gazetteer."""

    def __init__(self, gazetteer: dict[str, dict[str, list[str]]]) -> None:
        # Build a lowercased alias -> (type, canonical) lookup.
        self._lookup: dict[str, tuple[str, str]] = {}
        for type_, names in gazetteer.items():
            for canonical, aliases in names.items():
                for alias in [canonical, *aliases]:
                    self._lookup[alias.lower()] = (type_, canonical)

    def extract(self, text: str) -> list[Entity]:
        found: dict[str, Entity] = {}
        low = text.lower()
        for alias, (type_, canonical) in self._lookup.items():
            if re.search(rf"\b{re.escape(alias)}\b", low):
                eid = entity_id(type_, canonical)
                found[eid] = Entity(id=eid, type=type_, name=canonical)
        return list(found.values())


class LLMExtractor:
    """Schema-guided covenant/relationship extraction via an injectable LLM client."""

    _PROMPT = (
        "Extract a credit knowledge graph from the passage. Return ONLY JSON:\n"
        '{{"entities": [{{"type": "Issuer|Instrument|Sponsor|Covenant|Metric|Rating|Event", '
        '"name": "..."}}], "relations": [{{"source": "name", "type": '
        '"issues|ranks_above|owned_by|has_covenant|modifies|rated_as", "target": "name"}}]}}\n'
        "Only include entities and relations the passage supports.\n\nPassage:\n{passage}"
    )

    def __init__(self, llm: Any | None = None) -> None:
        if llm is None:
            from frag.rag.openrouter_client import OpenRouterClient

            llm = OpenRouterClient("EXTRACT_MODEL")
        self.llm = llm

    def extract(self, text: str) -> tuple[list[Entity], list[Relation]]:
        raw = self.llm.generate(self._PROMPT.format(passage=text[:6000]))
        try:
            data = json.loads(raw)
        except Exception:
            return [], []
        return _parse_entities(data.get("entities", [])), _parse_relations(
            data.get("relations", [])
        )


def _parse_entities(items: list[dict]) -> list[Entity]:
    out: dict[str, Entity] = {}
    for it in items:
        type_, name = str(it.get("type", "")).strip(), str(it.get("name", "")).strip()
        if type_ and name:
            eid = entity_id(type_, name)
            out[eid] = Entity(id=eid, type=type_, name=name)
    return list(out.values())


def _parse_relations(items: list[dict]) -> list[Relation]:
    out: list[Relation] = []
    for it in items:
        src, type_, tgt = it.get("source"), it.get("type"), it.get("target")
        if src and type_ and tgt:
            out.append(
                Relation(source=str(src).strip(), type=str(type_).strip(), target=str(tgt).strip())
            )
    return out


def hybrid_extract(
    text: str,
    gazetteer: dict[str, dict[str, list[str]]] | None = None,
    llm: Any | None = None,
) -> tuple[list[Entity], list[Relation]]:
    """Run the gazetteer (always) and the LLM (if provided); merge and dedupe."""
    entities: dict[str, Entity] = {}
    relations: list[Relation] = []

    if gazetteer:
        for e in GazetteerExtractor(gazetteer).extract(text):
            entities[e.id] = e

    if llm is not None:
        llm_entities, llm_relations = LLMExtractor(llm).extract(text)
        for e in llm_entities:
            entities.setdefault(e.id, e)
        # Resolve relation endpoints (names) to entity ids where we can.
        by_name = {e.name.lower(): e.id for e in entities.values()}
        for r in llm_relations:
            src_id = by_name.get(r.source.lower(), entity_id("Unknown", r.source))
            tgt_id = by_name.get(r.target.lower(), entity_id("Unknown", r.target))
            relations.append(Relation(source=src_id, type=r.type, target=tgt_id, attrs=r.attrs))

    return list(entities.values()), relations
