"""Build the credit knowledge graph from a corpus JSONL via hybrid extraction.

    python scripts/build_graph.py --corpus data/corpus.jsonl --out data/graph.json \
        --gazetteer data/gazetteer.json

Gazetteer extraction runs offline; LLM extraction runs when OPENROUTER_API_KEY is
set (EXTRACT_MODEL). No AWS.
"""

from __future__ import annotations

import argparse
import json

from frag.kg.extract import hybrid_extract
from frag.kg.graph import PropertyGraph
from frag.utils.config import configure_runtime


def _load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    configure_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="data/graph.json")
    ap.add_argument("--gazetteer", default=None)
    ap.add_argument("--no-llm", action="store_true", help="gazetteer only (offline)")
    args = ap.parse_args()

    gazetteer = json.load(open(args.gazetteer, encoding="utf-8")) if args.gazetteer else None
    llm = None
    if not args.no_llm:
        from frag.rag.openrouter_client import OpenRouterClient

        llm = OpenRouterClient("EXTRACT_MODEL")

    graph = PropertyGraph()
    for rec in _load_jsonl(args.corpus):
        entities, relations = hybrid_extract(rec.get("text", ""), gazetteer=gazetteer, llm=llm)
        for e in entities:
            graph.add_entity(e)
        for r in relations:
            graph.add_relation(r)

    graph.save(args.out)
    print({"nodes": graph.g.number_of_nodes(), "edges": graph.g.number_of_edges(), "out": args.out})


if __name__ == "__main__":
    main()
