from __future__ import annotations

import os
import re

import requests

# OpenRouter exposes an OpenAI-compatible chat-completions API. A single client
# serves actor, critic, and synthetic-query generation; the model is chosen per
# role through an environment variable (e.g. ACTOR_MODEL, CRITIC_MODEL), mirroring
# the OllamaClient contract so Actor/Critic need no change beyond the default.
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"


class OpenRouterClient:
    """Minimal OpenRouter chat client exposing `generate(prompt) -> str`.

    The response is cleaned to a bare JSON object/array so downstream callers can
    `json.loads` it directly, matching the OllamaClient behaviour it replaces.
    Swapping the local 3B judge for a capable hosted model is what turns the
    critic score from a noisy, directional signal into a trustworthy one.
    """

    def __init__(self, model_env_key: str) -> None:
        self.base = os.getenv("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv(model_env_key) or os.getenv("OPENROUTER_MODEL", _DEFAULT_MODEL)
        self.timeout = int(os.getenv("OPENROUTER_TIMEOUT", "120"))

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

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", "0")),
            "response_format": {"type": "json_object"},
        }

        r = requests.post(
            f"{self.base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        raw = data["choices"][0]["message"]["content"] or ""
        return self._clean(raw)
