"""Build the credit knowledge graph from a corpus JSONL via hybrid extraction.

    python scripts/build_graph.py --corpus data/corpus_build/corpus.jsonl \
        --out data/graph.json --gazetteer data/gazetteer_credit.json --no-llm

Each record becomes a document node (the contract/filing); its text is chunked so
extraction covers the whole document, not just the first window. The gazetteer path
runs offline and links the document to every covenant/metric/event it contains
(has_covenant). The LLM path (OPENROUTER_API_KEY + EXTRACT_MODEL) adds issuer/
instrument relationships. No AWS.
"""

from __future__ import annotations

import argparse
import json
import os

from frag.kg.extract import hybrid_extract
from frag.kg.graph import PropertyGraph
from frag.kg.schema import Entity, Relation, entity_id
from frag.sources.chunking import chunk_text
from frag.utils.config import configure_runtime

# Node types the document links to via has_covenant when found in its text.
_CONTAINED = {"Covenant", "Metric", "Event"}


def _load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _doc_node(rec: dict) -> Entity:
    """A contract/filing as a graph node, typed from its source."""
    meta = rec.get("metadata", {})
    title = meta.get("title") or meta.get("doc_name") or rec.get("id", "document")
    ntype = "Instrument" if meta.get("source_type") == "legal_contract" else "Issuer"
    return Entity(entity_id(ntype, title), ntype, title, {"source_docs": [rec["id"]]})


def main():
    configure_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="data/graph.json")
    ap.add_argument("--gazetteer", default=None)
    ap.add_argument("--no-llm", action="store_true", help="gazetteer only (offline)")
    ap.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "1500")))
    ap.add_argument("--limit", type=int, default=None, help="cap records (for a quick pass)")
    args = ap.parse_args()

    gazetteer = json.load(open(args.gazetteer, encoding="utf-8")) if args.gazetteer else None
    llm = None
    if not args.no_llm:
        from frag.rag.openrouter_client import OpenRouterClient

        llm = OpenRouterClient("EXTRACT_MODEL")

    graph = PropertyGraph()
    records = _load_jsonl(args.corpus)
    if args.limit:
        records = records[: args.limit]

    for rec in records:
        doc = _doc_node(rec)
        graph.add_entity(doc)
        for chunk in chunk_text(rec.get("text", ""), chunk_size=args.chunk_size):
            entities, relations = hybrid_extract(
                chunk, gazetteer=gazetteer, llm=llm, doc_id=rec["id"]
            )
            for e in entities:
                graph.add_entity(e)
                # Link the document to the covenants/metrics/events it contains.
                if e.type in _CONTAINED:
                    graph.add_relation(
                        Relation(doc.id, "has_covenant", e.id, {"source_doc": rec["id"]})
                    )
            for r in relations:
                graph.add_relation(r)

    graph.save(args.out)
    print(
        {
            "docs": len(records),
            "nodes": graph.g.number_of_nodes(),
            "edges": graph.g.number_of_edges(),
            "out": args.out,
        }
    )


if __name__ == "__main__":
    main()
