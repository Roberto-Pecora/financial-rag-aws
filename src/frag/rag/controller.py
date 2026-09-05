from __future__ import annotations

import logging
import os
from typing import Any, Protocol

import frag.rag.store_qdrant as store_qdrant
from frag.agent.guardrails import screen_input, screen_output
from frag.rag.actor import Actor
from frag.rag.actor import _doc_id as actor_doc_id
from frag.rag.critic import Critic
from frag.rag.entity_filter import company_filter
from frag.rag.explain import ground_answer
from frag.rag.grader import RetrievalGrader, RetrievalRewriter, corrective_enabled

logger = logging.getLogger(__name__)


class DocumentStore(Protocol):
    def ingest(self, docs: list[dict[str, Any]]) -> int: ...

    def search(
        self, query: str, top_k: int = 8, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...

    def count(self) -> int: ...


def _critic_enabled() -> bool:
    """The critic is an optional risk gate, off by default.

    This project's thesis is retrieval quality (trained embeddings + reranker,
    scored by deterministic IR metrics), so the ablation matrix runs actor-only
    to keep the comparison clean and halve LLM cost per query. Set CRITIC=on to
    demonstrate the actor-critic abstention gate as a financial risk control.
    """
    return os.getenv("CRITIC", "off").strip().lower() in {"on", "1", "true", "yes"}


def make_default_store() -> DocumentStore:
    """Build the store selected by STORE_BACKEND (default: opensearch).

    OpenSearch is the project's real backend (managed AWS hybrid search); the
    Qdrant path is retained so the pipeline still runs locally with no AWS.
    Imports are local so selecting one backend never requires the other's deps.
    """
    backend = os.getenv("STORE_BACKEND", "opensearch").strip().lower()
    if backend == "graph":
        from frag.kg.graph import PropertyGraph
        from frag.kg.graph_rag import GraphRAGRetriever

        return GraphRAGRetriever(PropertyGraph.load(os.environ["GRAPH_PATH"]))
    if backend == "qdrant":
        base: DocumentStore = store_qdrant.QdrantStore()
    else:
        from frag.rag.store_opensearch import OpenSearchStore

        base = OpenSearchStore()

    # Optional third IR stage: wrap the store so hybrid candidates are reranked
    # by the trained cross-encoder before the actor sees them (RERANK=on).
    from frag.rag.reranker import CrossEncoderReranker, RerankingStore, rerank_enabled

    if rerank_enabled():
        return RerankingStore(base, CrossEncoderReranker())
    return base


class RagController:
    """Thin orchestrator: store retrieves, actor generates, optional critic gates."""

    def __init__(self, store: DocumentStore | None = None) -> None:
        self.store = store or make_default_store()
        self.actor = Actor()
        self.use_critic = _critic_enabled()
        self.critic = Critic() if self.use_critic else None
        self.min_score = float(os.getenv("CRITIC_MIN_SCORE", "0.8"))
        # Corrective retrieval (CRAG): built lazily, only if CORRECTIVE is on.
        self.corrective = corrective_enabled()
        self.grader = RetrievalGrader() if self.corrective else None
        self.rewriter = RetrievalRewriter() if self.corrective else None
        self.max_rewrites = int(os.getenv("CORRECTIVE_MAX_REWRITES", "2"))
        self.min_relevant = int(os.getenv("CORRECTIVE_MIN_RELEVANT", "1"))

    def _corrective_retrieve(self, query: str, top_k: int, metadata_filter) -> list[dict]:
        """Retrieve, grade against the query, and rewrite-and-retry if too few relevant."""
        contexts = self.store.search(query=query, top_k=top_k, metadata_filter=metadata_filter)
        if self.grader is None:
            return contexts
        kept = self.grader.keep_relevant(query, contexts)
        for _ in range(self.max_rewrites):
            if len(kept) >= self.min_relevant:
                break
            rewritten = self.rewriter.rewrite(query)
            logger.info("corrective retrieval: rewrote %r -> %r", query, rewritten)
            contexts = self.store.search(
                query=rewritten, top_k=top_k, metadata_filter=metadata_filter
            )
            kept = self.grader.keep_relevant(rewritten, contexts)
        return kept or contexts  # fall back to raw hits rather than starving the actor

    def _entity_filter(self, query: str) -> dict[str, str] | None:
        """A company filter derived from the query, if the store knows its companies."""
        lister = getattr(self.store, "list_companies", None)
        if lister is None:
            return None
        if not hasattr(self, "_companies"):
            try:
                self._companies = lister()
            except Exception:
                self._companies = []
        return company_filter(query, self._companies)

    def prompt_versions(self) -> dict[str, int | None]:
        """Active prompt versions, for logging alongside eval metrics (A/B testing)."""
        return {
            "actor_prompt_version": self.actor.prompt_version,
            "critic_prompt_version": self.critic.prompt_version if self.critic else None,
        }

    # ------------------------------------------------------------------
    # Raw store operations
    # ------------------------------------------------------------------

    def ingest(self, docs: list[dict[str, Any]]) -> int:
        return self.store.ingest(docs)

    def query(
        self, query: str, top_k: int = 8, metadata_filter: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return raw retrieval hits only (no LLM synthesis)."""
        hits = self.store.search(query=query, top_k=top_k, metadata_filter=metadata_filter)
        return {"query": query, "top_k": top_k, "results": hits}

    def count(self) -> int:
        return self.store.count()

    # ------------------------------------------------------------------
    # Full actor-critic pipeline (orchestration lives here, NOT in API)
    # ------------------------------------------------------------------

    def answer_with_critique(
        self,
        query: str,
        top_k: int = 8,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve -> Actor generates answer -> Critic scores -> return enriched payload.

        Returns
        -------
        {
            "query":        str,
            "status":       "accepted" | "abstained",
            "answer":       str,
            "citations":    list[str],
            "critic_score": float,
            "critic_notes": str,
            "results":      list[dict],   # raw retrieved docs for transparency
        }
        """
        # Input guardrail: block override/injection attempts before any LLM call.
        verdict = screen_input(query)
        if not verdict.allowed:
            return {
                "query": query,
                "status": "refused",
                "answer": "Request blocked by the input guardrail.",
                "citations": [],
                "critic_score": None,
                "critic_notes": verdict.reason,
                "results": [],
            }

        # Entity-aware retrieval: constrain to the query's company when it names one.
        metadata_filter = metadata_filter or self._entity_filter(query)
        contexts = self._corrective_retrieve(query, top_k, metadata_filter)

        actor_out = self.actor.act(query, contexts)

        # Actor-only path (default): abstention is the actor's own signal — no
        # evidence retrieved, or the actor reporting insufficient evidence.
        if self.critic is None:
            actor_abstained = not contexts or actor_out["answer"].strip().lower().startswith(
                "insufficient evidence"
            )
            return self._guard_output(
                {
                    "query": query,
                    "status": "abstained" if actor_abstained else "accepted",
                    "answer": actor_out["answer"],
                    "citations": actor_out["citations"],
                    "critic_score": None,
                    "critic_notes": "critic disabled (CRITIC=off)",
                    "results": contexts,
                },
                contexts,
            )

        # Optional risk gate: a capable critic scores the draft and can veto it.
        critic_out = self.critic.critique(
            query, contexts, actor_out["answer"], actor_out["citations"]
        )
        accepted = critic_out["score"] >= self.min_score
        return self._guard_output(
            {
                "query": query,
                "status": "accepted" if accepted else "abstained",
                "answer": actor_out["answer"],
                "citations": actor_out["citations"],
                "critic_score": critic_out["score"],
                "critic_notes": critic_out["notes"],
                "results": contexts,
            },
            contexts,
        )

    def _guard_output(self, payload: dict[str, Any], contexts: list) -> dict[str, Any]:
        """Ground the answer, then downgrade to abstained if it fails the output guardrail."""
        # Explainability: which retrieved passage supports each fact in the answer.
        payload["grounding"] = ground_answer(payload["answer"], contexts, actor_doc_id)
        if payload["status"] != "accepted":
            return payload
        labels = {actor_doc_id(c, i) for i, c in enumerate(contexts)}
        verdict = screen_output(payload["answer"], payload["citations"], labels)
        if not verdict.allowed:
            payload["status"] = "abstained"
            payload["answer"] = "Answer withheld by the output guardrail."
            note = payload.get("critic_notes") or ""
            payload["critic_notes"] = f"{note} [guardrail: {verdict.reason}]".strip()
        return payload
