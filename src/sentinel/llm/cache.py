"""Response cache, free-tier throttle and provider fallback, as provider wrappers.

Both the cache and the call log live in one SQLite file, so quota accounting survives
restarts and a repeated evaluation run costs no API calls.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sentinel.common.logging import get_logger
from sentinel.llm.base import LLMError, LLMProvider, LLMResponse, QuotaError

log = get_logger(__name__)

DAY = 86_400.0


class _Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT, ts REAL)"
            )
            self._db.execute("CREATE TABLE IF NOT EXISTS calls (provider TEXT, ts REAL)")
            self._db.execute("CREATE INDEX IF NOT EXISTS calls_ts ON calls (provider, ts)")

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return None if row is None else dict(json.loads(row[0]))

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )

    def call_times(self, provider: str, since: float) -> list[float]:
        with self._lock:
            rows = self._db.execute(
                "SELECT ts FROM calls WHERE provider = ? AND ts >= ? ORDER BY ts",
                (provider, since),
            ).fetchall()
        return [float(r[0]) for r in rows]

    def record_call(self, provider: str, ts: float) -> None:
        with self._lock:
            self._db.execute("INSERT INTO calls VALUES (?, ?)", (provider, ts))
            self._db.execute("DELETE FROM calls WHERE ts < ?", (ts - 2 * DAY,))


_stores: dict[Path, _Store] = {}


def store(path: Path) -> _Store:
    path = path.resolve()
    if path not in _stores:
        _stores[path] = _Store(path)
    return _stores[path]


def cache_key(provider: LLMProvider, system: str, prompt: str, schema: Any, t: float) -> str:
    blob = json.dumps(
        [provider.name, provider.model, system, prompt, schema, t], sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedProvider:
    """Answers identical requests from the cache (deterministic at temperature 0)."""

    def __init__(self, inner: LLMProvider, path: Path) -> None:
        self.inner, self.name, self.model, self.local = inner, inner.name, inner.model, inner.local
        self._store = store(path)
        self.hits = self.misses = 0

    def generate(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        key = cache_key(self.inner, system, prompt, schema, temperature)
        hit = self._store.get(key)
        if hit is not None:
            self.hits += 1
            return LLMResponse(**{**hit, "cached": True, "seconds": 0.0})
        self.misses += 1
        resp = self.inner.generate(system, prompt, schema, temperature)
        self._store.put(key, asdict(resp))
        return resp


class ThrottledProvider:
    """Keeps a provider under a requests-per-minute and requests-per-day budget.

    Waits (up to `max_wait` seconds) for the minute window; raises QuotaError when the
    daily budget is spent or the wait would be longer, so the fallback chain moves on.
    """

    def __init__(
        self,
        inner: LLMProvider,
        path: Path,
        rpm: int,
        rpd: int,
        max_wait: float = 65.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner, self.name, self.model, self.local = inner, inner.name, inner.model, inner.local
        self.bucket = f"{inner.name}:{inner.model}"  # free-tier quotas are per model
        self._store = store(path)
        self.rpm, self.rpd, self.max_wait = rpm, rpd, max_wait
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()

    def remaining_today(self) -> int:
        return self.rpd - len(self._store.call_times(self.bucket, self._clock() - DAY))

    def _acquire(self) -> None:
        with self._lock:
            now = self._clock()
            if self.remaining_today() <= 0:
                raise QuotaError(f"{self.name}: daily budget of {self.rpd} requests spent")
            recent = self._store.call_times(self.bucket, now - 60)
            if len(recent) >= self.rpm:
                wait = recent[len(recent) - self.rpm] + 60 - now + 0.05
                if wait > self.max_wait:
                    raise QuotaError(f"{self.name}: would wait {wait:.0f}s for rate limit")
                log.info("%s: rate limit reached, waiting %.1fs", self.name, wait)
                self._sleep(wait)
                now = self._clock()
            self._store.record_call(self.bucket, now)

    def generate(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self._acquire()
        try:
            return self.inner.generate(system, prompt, schema, temperature)
        except QuotaError:
            # The server says the quota is gone (limits vary by account): stop for today.
            now = self._clock()
            for _ in range(max(0, self.remaining_today())):
                self._store.record_call(self.bucket, now)
            raise


class FallbackProvider:
    """Tries each provider in order; moves on after any LLMError (quota, network, bad reply)."""

    def __init__(self, providers: Sequence[LLMProvider]) -> None:
        if not providers:
            raise ValueError("no LLM providers configured")
        self.providers = list(providers)
        self.name = "+".join(p.name for p in self.providers)
        self.model = self.providers[0].model
        self.local = all(p.local for p in self.providers)

    def generate(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        errors = []
        for p in self.providers:
            try:
                return p.generate(system, prompt, schema, temperature)
            except LLMError as exc:
                log.warning("%s failed (%s); trying the next provider", p.name, exc)
                errors.append(f"{p.name}: {exc}")
        raise LLMError("all LLM providers failed: " + " | ".join(errors))
