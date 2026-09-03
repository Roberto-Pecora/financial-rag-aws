"""Session-scoped stubs so tests never need live Qdrant or a hosted LLM.

PATCHING STRATEGY
-----------------
autouse fixtures run per-test, AFTER all modules are imported.  But
`frag.api.main` constructs RagController() at module level, which can call
OpenRouterClient() and QdrantStore() at *import time* - before any fixture
can run.

Fix: use a session-scoped autouse fixture that patches both classes at
the CLASS level on their source modules before the first test is
collected.  Per-test fixtures then remain as a safety net.
"""

from __future__ import annotations

import os

import pytest

# Neutralise experiment-config env vars so a developer's local .env (which is
# loaded via frag.utils.config at import time) can't leak retrieval mode or
# chunking strategy into the test run and make it non-hermetic. CRITIC is forced
# off so the default actor-only path is what gets exercised unless a test opts in.
for _var in ("RETRIEVAL_MODE", "CHUNK_STRATEGY", "CHUNK_SIZE", "CHUNK_OVERLAP"):
    os.environ.pop(_var, None)
os.environ["CRITIC"] = "off"

# ---------------------------------------------------------------------------
# _StubStore  (no-dep in-memory DocumentStore)
# ---------------------------------------------------------------------------


class _StubStore:
    def __init__(self, *args, **kwargs):
        self._docs: list = []

    def ingest(self, docs):
        self._docs.extend(docs)
        return len(docs)

    def search(self, query, top_k=8, metadata_filter=None):
        return [
            {"text": d["text"], "metadata": d.get("metadata", {}), "score": 1.0}
            for d in self._docs
            if query.lower() in d["text"].lower()
        ][:top_k]

    def count(self):
        return len(self._docs)


# ---------------------------------------------------------------------------
# _StubLLMClient  (never makes a network call)
# ---------------------------------------------------------------------------

_ACTOR_RESPONSE = '{"answer":"stub answer","citations":["doc-1"]}'
_CRITIC_RESPONSE = (
    '{"overall_score":0.9,"faithfulness_score":0.95,'
    '"completeness_score":0.88,"citation_score":0.87,"issues":[]}'
)

_CRITIC_MARKERS = (
    "faithfulness_score",
    "citation_score",
    "completeness_score",
    "strict financial QA critic",
)


class _StubLLMClient:
    """Safe no-op LLM client.  __init__ accepts any kwargs without raising."""

    def __init__(self, *args, **kwargs):  # accepts model_env_key or anything
        pass

    def generate(self, prompt: str) -> str:
        if any(marker in prompt for marker in _CRITIC_MARKERS):
            return _CRITIC_RESPONSE
        return _ACTOR_RESPONSE


# ---------------------------------------------------------------------------
# SESSION-SCOPED patch  — runs once, before any module is imported by tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _patch_services_session():
    """Patch QdrantStore and OpenRouterClient at class level for the session.

    This runs before any test module is imported, so module-level
    RagController() construction in frag.api.main never sees the real classes.
    """
    import frag.rag.actor as actor_mod
    import frag.rag.controller as ctrl
    import frag.rag.critic as critic_mod
    import frag.rag.openrouter_client as orc
    import frag.rag.store_qdrant as sq

    # Patch at the source so every subsequent import gets the stub
    sq.QdrantStore = _StubStore
    orc.OpenRouterClient = _StubLLMClient
    actor_mod.OpenRouterClient = _StubLLMClient
    critic_mod.OpenRouterClient = _StubLLMClient

    # Also fix the reference already held by the controller module
    ctrl.store_qdrant = type("_mod", (), {"QdrantStore": _StubStore})()
    # The controller's default store now comes from a backend factory; stub it
    # so RagController() with no injected store never opens a live backend.
    ctrl.make_default_store = lambda: _StubStore()

    yield  # tests run here


# ---------------------------------------------------------------------------
# PER-TEST monkeypatch fixtures  (safety net; also resets between tests)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_qdrant(monkeypatch):
    import frag.rag.store_qdrant as sq

    monkeypatch.setattr(sq, "QdrantStore", _StubStore)
    import frag.rag.controller as ctrl

    monkeypatch.setattr(ctrl, "store_qdrant", type("_mod", (), {"QdrantStore": _StubStore})())
    monkeypatch.setattr(ctrl, "make_default_store", lambda: _StubStore())


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    import frag.rag.openrouter_client as orc

    monkeypatch.setattr(orc, "OpenRouterClient", _StubLLMClient)
    import frag.rag.actor as actor_mod

    monkeypatch.setattr(actor_mod, "OpenRouterClient", _StubLLMClient)
    import frag.rag.critic as critic_mod

    monkeypatch.setattr(critic_mod, "OpenRouterClient", _StubLLMClient)
