"""Local property graph over networkx, with entity resolution and JSON persistence."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
from rapidfuzz import fuzz, process

from frag.kg.schema import Entity, Relation


class PropertyGraph:
    """A credit graph: typed entity nodes and typed relation edges."""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    def add_entity(self, e: Entity) -> None:
        # Idempotent by id; merge attrs on repeat, unioning provenance.
        if self.g.has_node(e.id):
            attrs = self.g.nodes[e.id]["attrs"]
            merged_docs = {*attrs.get("source_docs", []), *e.attrs.get("source_docs", [])}
            attrs.update(e.attrs)
            if merged_docs:
                attrs["source_docs"] = sorted(merged_docs)
        else:
            self.g.add_node(e.id, type=e.type, name=e.name, attrs=dict(e.attrs))

    def add_relation(self, r: Relation) -> None:
        self.g.add_edge(r.source, r.target, key=r.type, type=r.type, attrs=dict(r.attrs))

    def resolve(self, name: str, threshold: int = 80) -> str | None:
        """Find an entity id by exact then fuzzy name match; None if nothing close."""
        names = {nid: data["name"] for nid, data in self.g.nodes(data=True)}
        for nid, nm in names.items():
            if nm.lower() == name.lower():
                return nid
        if not names:
            return None
        match = process.extractOne(name, names, scorer=fuzz.WRatio)
        return match[2] if match and match[1] >= threshold else None

    def provenance(self, entity_id: str) -> list[str]:
        """Source docs an entity was extracted from (for citations)."""
        if entity_id not in self.g:
            return []
        return list(self.g.nodes[entity_id]["attrs"].get("source_docs", []))

    def neighbours(self, entity_id: str, depth: int = 1) -> set[str]:
        if entity_id not in self.g:
            return set()
        und = self.g.to_undirected(as_view=True)
        return set(nx.single_source_shortest_path_length(und, entity_id, cutoff=depth)) - {
            entity_id
        }

    def subgraph_text(self, entity_id: str, depth: int = 1) -> str:
        """Serialise the entity and its neighbourhood as cited, readable lines."""
        if entity_id not in self.g:
            return ""
        nodes = {entity_id} | self.neighbours(entity_id, depth)
        lines = []
        for u, v, data in self.g.edges(nodes, data=True):
            if u in nodes and v in nodes:
                lines.append(f"{self.g.nodes[u]['name']} {data['type']} {self.g.nodes[v]['name']}")
        return "\n".join(sorted(set(lines)))

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"id": n, "type": d["type"], "name": d["name"], "attrs": d["attrs"]}
                for n, d in self.g.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "type": d["type"], "attrs": d["attrs"]}
                for u, v, d in self.g.edges(data=True)
            ],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PropertyGraph:
        pg = cls()
        for n in data.get("nodes", []):
            pg.add_entity(
                Entity(id=n["id"], type=n["type"], name=n["name"], attrs=n.get("attrs", {}))
            )
        for e in data.get("edges", []):
            pg.add_relation(Relation(source=e["source"], type=e["type"], target=e["target"]))
        return pg

    @classmethod
    def load(cls, path: str) -> PropertyGraph:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
