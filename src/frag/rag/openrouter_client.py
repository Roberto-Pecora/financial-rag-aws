from __future__ import annotations

import os
import re
from typing import Any

import requests

from frag.eval.pricing import cost_from_usage
from frag.eval.trace import Timer
from frag.utils.http_retry import request_with_retry

# OpenAI-compatible client; model chosen per role via env (ACTOR_MODEL, etc.).
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"


def _structured_output_enabled() -> bool:
    return os.getenv("LLM_STRUCTURED_OUTPUT", "off").strip().lower() in {"on", "1", "true", "yes"}


class OpenRouterClient:
    """OpenRouter chat client exposing `generate(prompt) -> str` (cleaned JSON).

    `response_model` (a Pydantic model) drives structured output: when
    LLM_STRUCTURED_OUTPUT is on, its JSON schema constrains generation; otherwise
    the request just asks for a JSON object.
    """

    def __init__(self, model_env_key: str, response_model: Any | None = None) -> None:
        self.base = os.getenv("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv(model_env_key) or os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL)
        self.timeout = int(os.getenv("OPENROUTER_TIMEOUT", "120"))
        self.max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "4"))
        self.retry_base_delay = float(os.getenv("OPENROUTER_RETRY_BASE_DELAY", "2"))
        self.response_model = response_model
        # Per-call telemetry, set after each generate() for latency/cost eval.
        self.last_latency_s = 0.0
        self.last_usage: dict | None = None
        self.last_cost = 0.0

    def _response_format(self) -> dict[str, Any]:
        if self.response_model is not None and _structured_output_enabled():
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_model.__name__.lower(),
                    "schema": self.response_model.model_json_schema(),
                },
            }
        return {"type": "json_object"}

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0")),
            "response_format": self._response_format(),
        }

    @staticmethod
    def _clean(raw: str) -> str:
        """Strip markdown fences and extract the first JSON object/array."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text.strip()).strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return m.group(0) if m else text

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional attribution headers OpenRouter recommends; harmless if unset.
        if referer := os.getenv("OPENROUTER_REFERER"):
            headers["HTTP-Referer"] = referer
        if title := os.getenv("OPENROUTER_TITLE"):
            headers["X-Title"] = title
        return headers

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to chat/completions with retry; record telemetry; return the JSON."""
        headers = self._headers()
        url = f"{self.base}/chat/completions"
        with Timer() as t:
            r = request_with_retry(
                lambda: requests.post(url, headers=headers, json=payload, timeout=self.timeout),
                max_retries=self.max_retries,
                base_delay=self.retry_base_delay,
                label=f"openrouter[{self.model}]",
            )
            data = r.json()
        self.last_latency_s = t.elapsed
        self.last_usage = data.get("usage")
        self.last_cost = cost_from_usage(self.model, self.last_usage)
        return data

    def generate(self, prompt: str) -> str:
        data = self._post(self._build_payload(prompt))
        raw = data["choices"][0]["message"]["content"] or ""
        return self._clean(raw)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict[str, Any]:
        """Multi-message call with native function-calling; returns the assistant message.

        The returned dict has `content` (str | None) and `tool_calls` (a list of
        {id, name, arguments} — arguments is the raw JSON string the model emitted).
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0")),
        }
        if tools:
            payload["tools"] = tools
        message = self._post(payload)["choices"][0]["message"]
        calls = [
            {
                "id": c.get("id"),
                "name": c.get("function", {}).get("name"),
                "arguments": c.get("function", {}).get("arguments", "{}"),
            }
            for c in message.get("tool_calls") or []
        ]
        return {"content": message.get("content"), "tool_calls": calls}
