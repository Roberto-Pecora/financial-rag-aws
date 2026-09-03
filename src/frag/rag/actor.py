from __future__ import annotations

import json
from typing import Protocol

from frag.rag.openrouter_client import OpenRouterClient


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


def _doc_id(ctx: dict, idx: int) -> str:
    """Resolve a stable identifier from a context dict returned by QdrantStore.search().

    The store returns {text, metadata, score}. The original doc id lives inside
    metadata under 'doc_id', 'id', or 'ticker' — fall back to a positional label.
    """
    meta = ctx.get("metadata") or {}
    return meta.get("doc_id") or meta.get("id") or meta.get("ticker") or f"doc-{idx + 1}"


class Actor:
    def __init__(self, llm_client: LLMClient | None = None):
        self.client = llm_client or OpenRouterClient("ACTOR_MODEL")

    def build_prompt(self, query: str, contexts: list) -> str:
        evidence = "\n\n".join([f"[{_doc_id(c, i)}] {c['text']}" for i, c in enumerate(contexts)])
        return (
            "You are an investment research assistant.\n"
            "Answer the question using ONLY the evidence provided.\n"
            "Do not use outside knowledge.\n"
            "Do not mention documents that do not support the claim.\n"
            "Do not add market commentary unless explicitly supported by the evidence.\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            '{ "answer": string, "citations": [string, ...] }\n'
            "Rules:\n"
            "- answer must be one concise paragraph or sentence.\n"
            "- citations must be the doc label strings shown in the evidence (e.g. doc-1).\n"
            "- cite only documents that directly support the answer.\n"
            "- if evidence is insufficient, answer must be exactly: "
            "Insufficient evidence retrieved.\n"
            f"Question: {query}\n"
            f"Evidence:\n{evidence}"
        )

    def act(self, query: str, contexts: list) -> dict:
        if not contexts:
            raw = '{"answer":"Insufficient evidence retrieved.","citations":[]}'
            return {"answer": "Insufficient evidence retrieved.", "citations": [], "raw": raw}

        prompt = self.build_prompt(query, contexts)
        raw = self.client.generate(prompt)

        try:
            data = json.loads(raw)
            answer = str(data.get("answer", "")).strip()
            citations = data.get("citations", [])
            if not isinstance(citations, list):
                citations = []
            citations = [str(c).strip() for c in citations]
            return {"answer": answer, "citations": citations, "raw": raw}
        except Exception:
            return {
                "answer": "Insufficient evidence retrieved.",
                "citations": [],
                "raw": raw,
            }
