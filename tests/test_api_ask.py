"""The /v1/ask route delegates to the orchestrator; the UI exposes the agent toggle."""

from __future__ import annotations

from fastapi.testclient import TestClient

from frag.api import main


class _FakeOrchestrator:
    def answer(self, question, history=None):
        return {"route": "multi_hop", "status": "answered", "answer": f"agent:{question}"}


def _client(monkeypatch):
    monkeypatch.setattr(main, "get_orchestrator", lambda: _FakeOrchestrator())
    return TestClient(main.app)


def test_ask_route_returns_orchestrator_result(monkeypatch):
    resp = _client(monkeypatch).post("/v1/ask", json={"question": "compare Acme and Beta"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "multi_hop" and body["answer"] == "agent:compare Acme and Beta"


def test_ask_accepts_history(monkeypatch):
    resp = _client(monkeypatch).post(
        "/v1/ask",
        json={"question": "and its margin?", "history": [{"role": "user", "content": "revenue?"}]},
    )
    assert resp.status_code == 200


def test_ui_has_agent_toggle():
    html = TestClient(main.app).get("/").text
    assert 'id="mode"' in html and ">Agent<" in html
    assert "/v1/ask" in html  # the submit handler can reach the agent route
