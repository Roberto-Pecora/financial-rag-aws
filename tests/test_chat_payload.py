"""OpenRouterClient.chat: payload shape and tool_calls parsing, no network."""

from __future__ import annotations

from frag.rag.openrouter_client import OpenRouterClient


def _client(monkeypatch, response):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    c = OpenRouterClient("AGENT_MODEL")
    c._post = lambda payload: (setattr(c, "_seen", payload) or response)  # capture + fake
    return c


def test_chat_sends_messages_and_tools(monkeypatch):
    resp = {"choices": [{"message": {"content": "hi", "tool_calls": []}}]}
    c = _client(monkeypatch, resp)
    tools = [{"type": "function", "function": {"name": "retrieve"}}]
    out = c.chat([{"role": "user", "content": "q"}], tools=tools)
    assert c._seen["messages"][0]["content"] == "q"
    assert c._seen["tools"] == tools
    assert out == {"content": "hi", "tool_calls": []}


def test_chat_omits_tools_when_none(monkeypatch):
    c = _client(monkeypatch, {"choices": [{"message": {"content": "x"}}]})
    c.chat([{"role": "user", "content": "q"}])
    assert "tools" not in c._seen


def test_chat_parses_tool_calls(monkeypatch):
    resp = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "retrieve", "arguments": '{"query":"x"}'}}
                    ],
                }
            }
        ]
    }
    c = _client(monkeypatch, resp)
    out = c.chat([{"role": "user", "content": "q"}])
    assert out["tool_calls"] == [{"id": "c1", "name": "retrieve", "arguments": '{"query":"x"}'}]
