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

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Optional attribution headers OpenRouter recommends; harmless if unset.
        referer = os.getenv("OPENROUTER_REFERER")
        title = os.getenv("OPENROUTER_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        payload = self._build_payload(prompt)

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
        raw = data["choices"][0]["message"]["content"] or ""
        return self._clean(raw)
