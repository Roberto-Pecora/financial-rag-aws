"""GraphRAG retriever: resolve query to entities, expand the sub-graph as context.

Implements the store `search()` contract ({text, metadata, score}) so actor,
critic and the eval harness are unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from frag.kg.graph import PropertyGraph


class GraphRAGRetriever:
    def __init__(self, graph: PropertyGraph, depth: int = 1) -> None:
        self.graph = graph
        self.depth = depth

    def _link(self, query: str) -> list[str]:
        """Link query terms to graph entity ids (multi-word node names first)."""
        linked: list[str] = []
        names = [(nid, d["name"]) for nid, d in self.graph.g.nodes(data=True)]
        for nid, name in sorted(names, key=lambda x: -len(x[1])):
            if re.search(rf"\b{re.escape(name.lower())}\b", query.lower()) and nid not in linked:
                linked.append(nid)
        return linked

    def search(
        self, query: str, top_k: int = 8, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not query:
            return []
        results: list[dict[str, Any]] = []
        for rank, eid in enumerate(self._link(query)):
            text = self.graph.subgraph_text(eid, self.depth)
            if not text:
                continue
            results.append(
                {
                    "text": text,
                    "metadata": {
                        "doc_id": eid,
                        "entity": self.graph.g.nodes[eid]["name"],
                        "source": "graph",
                        "source_docs": self.graph.provenance(eid),
                    },
                    "score": 1.0 / (rank + 1),
                }
            )
        return results[:top_k]

    def count(self) -> int:
        return self.graph.g.number_of_nodes()
