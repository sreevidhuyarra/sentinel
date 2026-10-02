"""A local model served by Ollama (offline fallback, and the evaluation judge)."""

from __future__ import annotations

import time
from typing import Any

import httpx

from sentinel.llm.base import LLMError, LLMResponse, inline_refs


class OllamaProvider:
    name = "ollama"
    local = True

    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434",
        timeout: float = 600.0,
        num_ctx: int = 8192,  # Ollama's default 4k truncates investigation prompts
    ) -> None:
        self.model, self.num_ctx = model, num_ctx
        self._url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def available(self) -> bool:
        try:
            tags = self._client.get(f"{self._url}/api/tags", timeout=3).json()
        except (httpx.HTTPError, ValueError):
            return False
        names = {m["name"] for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def generate(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature, "seed": 0, "num_ctx": self.num_ctx},
        }
        if schema is not None:
            body["format"] = inline_refs(schema)
        t0 = time.perf_counter()
        try:
            r = self._client.post(f"{self._url}/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed (is `ollama serve` running?): {exc}") from exc
        if r.status_code >= 400:
            raise LLMError(f"ollama HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        return LLMResponse(
            text=data["message"]["content"],
            provider=self.name,
            model=self.model,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            seconds=time.perf_counter() - t0,
        )
