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
