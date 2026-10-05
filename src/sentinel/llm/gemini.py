"""Google Gemini over its REST API (JSON mode via responseJsonSchema)."""

from __future__ import annotations

import time
from typing import Any

import httpx

from sentinel.llm.base import LLMError, LLMResponse, QuotaError, inline_refs

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_QUOTA_WAIT = 90.0  # seconds; longer retry delays are treated as "come back later"


def quota_info(r: httpx.Response) -> dict[str, Any]:
    """Which quota a 429 hit (per-minute vs per-day) and how long Google says to wait."""
    try:
        err = r.json().get("error")
    except ValueError:
        err = None
    details = err.get("details", []) if isinstance(err, dict) else []
    ids: list[str] = []
    retry: float | None = None
    for d in details:
        kind = d.get("@type", "")
        if kind.endswith("QuotaFailure"):
            ids += [str(v.get("quotaId", "")) for v in d.get("violations", [])]
        elif kind.endswith("RetryInfo"):
            try:
                retry = float(str(d.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                retry = None
    return {
        "quota_ids": [i for i in ids if i],
        "daily": any("PerDay" in i for i in ids),
        "retry_s": retry,
    }


class GeminiProvider:
    name = "gemini"
    local = False  # free-tier prompts may be reviewed by Google: send only redacted text

    # Transient server-side errors ("high demand"); retried with backoff, unlike 429 quota.
    RETRY_STATUS = frozenset({500, 502, 503, 504})

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        retries: int = 4,
        backoff: float = 10.0,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.model, self.retries, self.backoff = model, retries, backoff
        self._client = httpx.Client(timeout=timeout, headers={"x-goog-api-key": api_key})

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        for attempt in range(self.retries + 1):
            delay = self.backoff * 2**attempt  # 10, 20, 40, 80 s
            try:
                r = self._client.post(API.format(model=self.model), json=body)
            except httpx.TransportError as exc:  # timeouts, dropped connections
                if attempt == self.retries:
                    raise LLMError(f"gemini request failed: {exc}") from exc
            else:
                if r.status_code == 429:
                    # Per-minute limits come with a short retry delay: wait it out.
                    # Per-day limits (or long delays) go back to the caller as quota.
                    info = quota_info(r)
                    wait = info["retry_s"]
                    if (
                        info["daily"]
                        or wait is None
                        or wait > MAX_QUOTA_WAIT
                        or attempt == self.retries
                    ):
                        return r
                    delay = wait + 1.0
                elif r.status_code not in self.RETRY_STATUS or attempt == self.retries:
                    return r
            time.sleep(delay)
        raise AssertionError("unreachable")

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
            r = self._post(body)
        except httpx.HTTPError as exc:
            raise LLMError(f"gemini request failed: {exc}") from exc
        if r.status_code == 429:
            info = quota_info(r)
            raise QuotaError(
                f"gemini quota ({', '.join(info['quota_ids']) or 'unspecified'}; "
                f"retry after {info['retry_s']} s)",
                daily=info["daily"],
            )
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
