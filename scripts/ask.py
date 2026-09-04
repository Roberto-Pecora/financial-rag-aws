"""Ask the agent a multi-hop question over the local corpus + knowledge graph.

    python scripts/ask.py "which agreements have a change-of-control covenant?" \
        --corpus data/corpus_build/corpus.jsonl --graph data/corpus_build/graph.json

Retrieval and the graph run locally (no AWS); only the agent's LLM calls hit
OpenRouter (AGENT_MODEL). Prints the answer plus a trace of tools, cost and status.
"""

from __future__ import annotations

import argparse
import json
import os

from frag.agent.loop import AgentLoop
from frag.agent.tools import (
    ToolRegistry,
    make_calc_tool,
    make_graph_lookup_tool,
    make_retrieve_tool,
)
from frag.rag import prompts
from frag.utils.config import configure_runtime


def _load_corpus(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    configure_runtime()
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--corpus", default="data/corpus_build/corpus.jsonl")
    ap.add_argument("--graph", default="data/corpus_build/graph.json")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max-turns", type=int, default=int(os.getenv("AGENT_MAX_TURNS", "6")))
    ap.add_argument(
        "--max-tool-calls", type=int, default=int(os.getenv("AGENT_MAX_TOOL_CALLS", "8"))
    )
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    from frag.eval.local_retrieval import LocalHybridIndex
    from frag.kg.graph import PropertyGraph
    from frag.kg.graph_rag import GraphRAGRetriever
    from frag.rag.openrouter_client import OpenRouterClient

    store = LocalHybridIndex(_load_corpus(args.corpus), SentenceTransformer(args.model))
    graph = GraphRAGRetriever(PropertyGraph.load(args.graph))
    registry = ToolRegistry(
        [make_retrieve_tool(store), make_graph_lookup_tool(graph), make_calc_tool()]
    )

    loop = AgentLoop(
        OpenRouterClient("AGENT_MODEL"),
        registry,
        prompts.get("agent").body,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
    )
    out = loop.run(args.question)

    print("\nAnswer:", out["answer"])
    print(
        f"\nStatus: {out['status']}  tool_calls: {out['tool_calls']}  "
        f"cost: ${out['total_cost']:.4f}"
    )
    print("Trace:")
    for step in out["trace"]:
        print(f"  turn {step['turn']}: {step.get('tool') or '(final)'}  {step.get('args', '')}")


if __name__ == "__main__":
    main()
