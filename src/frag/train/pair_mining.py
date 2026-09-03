"""Training-pair mining for the embedding finetune and the reranker.

The training signal is built from the corpus itself, no human labels:

  1. **Synthetic queries** - an LLM reads a passage and writes questions it
     answers (Doc2Query / InPars style). Each (query, source passage) is a
     positive pair.
  2. **Hard negatives** - for each query, a cheap in-memory retriever surfaces
     passages that *look* relevant but are not the source. These lexically or
     semantically close distractors are what teach the embedding model and the
     cross-encoder to discriminate, rather than trivially separable random
     negatives.

Both collaborators are injected - the LLM client and the retriever - so this
orchestration is unit-tested with fakes and needs neither an API key nor a GPU.
`InMemoryRetriever` (BM25) and `SyntheticQueryGenerator` (OpenRouter) are the
concrete implementations, imported lazily.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

Record = dict[str, Any]


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[dict[str, Any]]: ...


class QueryGenerator(Protocol):
    def generate_queries(self, passage: str, n: int) -> list[str]: ...


def _rec_id(hit: dict[str, Any]) -> str:
    return hit.get("id") or (hit.get("metadata") or {}).get("doc_id") or ""


def mine_hard_negatives(
    query: str,
    positive_id: str,
    retriever: Retriever,
    n_negatives: int = 4,
    over_fetch: int | None = None,
) -> list[dict[str, str]]:
    """Return up to `n_negatives` high-ranked passages that are not the positive.

    Over-fetches so that removing the positive (and any duplicates) still leaves
    enough negatives. Each negative carries its id and text for the reranker.
    """
    over = over_fetch or (n_negatives + 5)
    negatives: list[dict[str, str]] = []
    seen: set[str] = {positive_id}
    for hit in retriever.search(query, top_k=over):
        hid = _rec_id(hit)
        if hid in seen:
            continue
        seen.add(hid)
        negatives.append({"id": hid, "text": hit.get("text", "")})
        if len(negatives) >= n_negatives:
            break
    return negatives


def build_training_pairs(
    corpus: list[Record],
    generator: QueryGenerator,
    retriever: Retriever,
    queries_per_passage: int = 2,
    n_negatives: int = 4,
    max_passages: int | None = None,
) -> list[dict[str, Any]]:
    """Mine (query, positive, negatives) triples from a corpus.

    One passage yields `queries_per_passage` queries; each query is paired with
    its source passage (positive) and `n_negatives` mined hard negatives.
    """
    pairs: list[dict[str, Any]] = []
    passages = corpus[:max_passages] if max_passages else corpus
    for rec in passages:
        pid = rec.get("id") or (rec.get("metadata") or {}).get("doc_id") or ""
        ptext = rec.get("text", "")
        if not ptext:
            continue
        for query in generator.generate_queries(ptext, queries_per_passage):
            query = query.strip()
            if not query:
                continue
            negatives = mine_hard_negatives(query, pid, retriever, n_negatives)
            pairs.append(
                {
                    "query": query,
                    "positive": ptext,
                    "positive_id": pid,
                    "negatives": negatives,
                }
            )
    return pairs


def write_pairs_jsonl(pairs: list[dict[str, Any]], path: str) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(pairs)


# --------------------------------------------------------------------------
# Concrete implementations (lazy heavy imports; kept out of unit tests)
# --------------------------------------------------------------------------


class SyntheticQueryGenerator:
    """Generate answerable questions from a passage via an OpenRouter model."""

    _PROMPT = (
        "You are building a retrieval training set for financial documents.\n"
        "Read the passage and write {n} short, specific questions that the "
        "passage answers. Prefer questions a financial analyst would ask about "
        "figures, trends, or risks. Do not answer them.\n"
        'Return ONLY JSON: {{"queries": ["...", "..."]}}\n\n'
        "Passage:\n{passage}"
    )

    def __init__(self, llm: Any | None = None) -> None:
        if llm is None:
            from frag.rag.openrouter_client import OpenRouterClient

            llm = OpenRouterClient("PAIRGEN_MODEL")
        self.llm = llm

    def generate_queries(self, passage: str, n: int = 2) -> list[str]:
        raw = self.llm.generate(self._PROMPT.format(n=n, passage=passage[:4000]))
        try:
            data = json.loads(raw)
            queries = data.get("queries", [])
            return [str(q).strip() for q in queries if str(q).strip()][:n]
        except Exception:
            return []


class InMemoryRetriever:
    """BM25 retrieval over an in-memory corpus, for hard-negative mining.

    BM25 is deliberate: lexically overlapping distractors are exactly the hard
    negatives that hurt dense retrieval, so mining them targets the weakness the
    finetune is meant to fix. No GPU, no external service.
    """

    def __init__(self, corpus: list[Record]) -> None:
        import re

        from rank_bm25 import BM25Okapi

        self._records = [r for r in corpus if r.get("text")]
        self._tokenize = lambda t: re.findall(r"[a-z0-9]+", t.lower())
        self._bm25 = BM25Okapi([self._tokenize(r["text"]) for r in self._records])

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        scores = self._bm25.get_scores(self._tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out = []
        for i in order:
            r = self._records[i]
            out.append(
                {
                    "id": r.get("id") or (r.get("metadata") or {}).get("doc_id") or "",
                    "text": r["text"],
                    "score": float(scores[i]),
                }
            )
        return out
