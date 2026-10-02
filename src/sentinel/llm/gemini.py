"""Google Gemini over its REST API (JSON mode via responseJsonSchema)."""

from __future__ import annotations

import time
from typing import Any

import httpx

from sentinel.llm.base import LLMError, LLMResponse, QuotaError, inline_refs

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiProvider:
    name = "gemini"
    local = False  # free-tier prompts may be reviewed by Google: send only redacted text

    def __init__(self, api_key: str, model: str, timeout: float = 60.0) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.model = model
        self._client = httpx.Client(timeout=timeout, headers={"x-goog-api-key": api_key})

    def generate(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        config: dict[str, Any] = {"temperature": temperature}
        if schema is not None:
            config["responseMimeType"] = "application/json"
            config["responseJsonSchema"] = inline_refs(schema)
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }
        t0 = time.perf_counter()
        try:
            r = self._client.post(API.format(model=self.model), json=body)
        except httpx.HTTPError as exc:
            raise LLMError(f"gemini request failed: {exc}") from exc
        if r.status_code == 429:
            raise QuotaError(f"gemini quota: {r.text[:200]}")
        if r.status_code >= 400:
            raise LLMError(f"gemini HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            cand = data["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"])
        except (KeyError, IndexError) as exc:
            reason = data.get("promptFeedback") or data.get("candidates", [{}])[0]
            raise LLMError(f"gemini returned no text: {reason}") from exc
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=int(usage.get("promptTokenCount", 0)),
            output_tokens=int(usage.get("candidatesTokenCount", 0)),
            seconds=time.perf_counter() - t0,
            meta={"finish_reason": cand.get("finishReason")},
        )
