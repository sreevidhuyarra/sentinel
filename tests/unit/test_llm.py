from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sentinel.llm.base import LLMError, LLMResponse, QuotaError, inline_refs, parse_json
from sentinel.llm.cache import CachedProvider, FallbackProvider, ThrottledProvider
from sentinel.llm.redact import Redactor


class Echo:
    """Test provider: answers with a counter so repeated calls are distinguishable."""

    local = True

    def __init__(self, name: str = "echo", fail: Exception | None = None) -> None:
        self.name, self.model, self.fail, self.calls = name, "m", fail, 0

    def generate(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return LLMResponse(text=f'{{"n": {self.calls}}}', provider=self.name, model=self.model)


def test_parse_json_tolerates_fences_and_prose() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_inline_refs_keeps_fields_named_title() -> None:
    schema = {
        "$defs": {"Item": {"title": "Item", "properties": {"title": {"type": "string"}}}},
        "properties": {
            "title": {"type": "string", "title": "Title"},
            "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
        },
        "title": "Draft",
    }
    flat = inline_refs(schema)
    assert "$defs" not in flat and "title" not in flat
    assert flat["properties"]["title"] == {"type": "string"}  # field kept, annotation dropped
    assert flat["properties"]["items"]["items"] == {"properties": {"title": {"type": "string"}}}


def test_cache_answers_repeated_requests(tmp_path: Path) -> None:
    inner = Echo()
    c = CachedProvider(inner, tmp_path / "c.sqlite")
    a, b = c.generate("s", "p"), c.generate("s", "p")
    assert (a.cached, b.cached, inner.calls) == (False, True, 1)
    assert b.json() == a.json()
    c.generate("s", "other prompt")
    assert inner.calls == 2


def test_throttle_waits_for_the_minute_window_and_stops_at_the_daily_budget(tmp_path: Path) -> None:
    now = [1000.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    t = ThrottledProvider(
        Echo(), tmp_path / "t.sqlite", rpm=2, rpd=3, clock=lambda: now[0], sleep=sleep
    )
    t.generate("s", "1")
    t.generate("s", "2")
    t.generate("s", "3")  # third in the same minute: waits ~60 s
    assert len(slept) == 1 and 59 < slept[0] <= 61
    with pytest.raises(QuotaError):
        t.generate("s", "4")  # daily budget of 3 spent


def test_throttle_raises_instead_of_waiting_too_long(tmp_path: Path) -> None:
    t = ThrottledProvider(
        Echo(), tmp_path / "t.sqlite", rpm=1, rpd=10, max_wait=5, clock=lambda: 0.0
    )
    t.generate("s", "1")
    with pytest.raises(QuotaError):
        t.generate("s", "2")


def test_fallback_moves_on_after_quota() -> None:
    first, second = Echo("gemini", fail=QuotaError("429")), Echo("ollama")
    f = FallbackProvider([first, second])
    assert f.generate("s", "p").provider == "ollama"
    with pytest.raises(LLMError):
        FallbackProvider([Echo("a", fail=LLMError("x")), Echo("b", fail=LLMError("y"))]).generate(
            "s", "p"
        )


def test_redactor_round_trip_is_stable() -> None:
    r = Redactor()
    text = "192.168.10.50 was hit by 205.174.165.73, then 192.168.10.50 again; mail bob@example.com"
    red = r.redact(text)
    assert "192.168" not in red and "example.com" not in red
    assert red.count("[ip-internal-1]") == 2 and "[ip-external-1]" in red and "[email-1]" in red
    assert r.restore(red) == text
    assert r.restore({"hosts": ["[ip-internal-1]"], "n": 3}) == {"hosts": ["192.168.10.50"], "n": 3}
    assert r.redact("port 8080, duration 5000000") == "port 8080, duration 5000000"


def test_gemini_retries_transient_errors_but_not_quota() -> None:
    import httpx

    from sentinel.llm.gemini import GeminiProvider

    ok = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}], "usageMetadata": {}}
    for statuses, expect in (([503, 503, 200], "ok"), ([429], "quota"), ([503] * 3, "error")):
        seen = list(statuses)

        def handler(_req: httpx.Request, seen: list[int] = seen) -> httpx.Response:
            code = seen.pop(0)
            return httpx.Response(code, json=ok if code == 200 else {"error": code})

        g = GeminiProvider("k", "m", retries=2, backoff=0.0)
        g._client = httpx.Client(transport=httpx.MockTransport(handler))
        if expect == "ok":
            assert g.generate("s", "p").json() == {"ok": True} and not seen
        elif expect == "quota":
            with pytest.raises(QuotaError):
                g.generate("s", "p")
        else:
            with pytest.raises(LLMError, match="503"):
                g.generate("s", "p")


def _quota_response(quota_id: str, retry: str | None) -> dict[str, Any]:
    details: list[dict[str, Any]] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id}],
        }
    ]
    if retry is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry})
    return {
        "error": {"code": 429, "message": "You exceeded your current quota", "details": details}
    }


def test_gemini_waits_out_per_minute_quota_and_flags_daily(tmp_path: Path) -> None:
    import httpx

    from sentinel.llm.gemini import GeminiProvider

    ok = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}], "usageMetadata": {}}
    minute = _quota_response("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "0s")
    day = _quota_response("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "3600s")

    def client(responses: list[tuple[int, dict[str, Any]]]) -> httpx.Client:
        def handler(_req: httpx.Request) -> httpx.Response:
            code, body = responses.pop(0)
            return httpx.Response(code, json=body)

        return httpx.Client(transport=httpx.MockTransport(handler))

    g = GeminiProvider("k", "m", retries=2, backoff=0.0)
    g._client = client([(429, minute), (200, ok)])
    assert g.generate("s", "p").json() == {"ok": True}  # waited, retried, answered

    g._client = client([(429, day)])
    with pytest.raises(QuotaError) as daily:
        g.generate("s", "p")
    assert daily.value.daily and "PerDay" in str(daily.value)

    # The throttle writes off the day only for a daily quota.
    for exc, spent in ((QuotaError("minute"), False), (QuotaError("day", daily=True), True)):
        t = ThrottledProvider(Echo(fail=exc), tmp_path / f"q{spent}.sqlite", rpm=50, rpd=10)
        with pytest.raises(QuotaError):
            t.generate("s", "p")
        assert (t.remaining_today() <= 0) is spent


def test_redactor_catches_ips_at_sentence_end_but_not_longer_numbers() -> None:
    r = Redactor()
    red = r.redact("Traffic from 172.16.0.1 hit 192.168.10.50. Version 1.2.3.4.5 is unrelated.")
    assert "172.16.0.1" not in red and "192.168.10.50" not in red
    assert "[ip-internal-2]." in red and "1.2.3.4.5" in red
