"""Public read-only demo: a small SQLite snapshot of the alert database, and the demo guards.

The demo serves the dashboard from a snapshot (no Postgres, Kafka or MLflow), scores emails
live, and turns off everything that writes, costs money or needs the copilot's models:
analyst feedback, live investigations (stored reports are still shown) and the live
robustness run. Scoring endpoints are rate-limited per client.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import Engine, insert, select, text

from sentinel.common.logging import get_logger
from sentinel.db.schema import alerts, drift_checks, init_db, make_engine, reports, retrain_runs

log = get_logger(__name__)

WINDOW_HOURS = 4  # the overview's longest window
DEDUP_KEY = ("ts", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "predicted_family")
BATCH = 5000

OFF = "Turned off in the public demo. Run Sentinel locally to use it (see the README)."
BLOCKED = (("POST", "/feedback/"), ("POST", "/robustness/run"))
LIMITED = ("/score/", "/copilot/")
RATE_PER_MINUTE = 20


# -- snapshot ------------------------------------------------------------------------------


def snapshot(src: Engine, out: Path) -> dict[str, int]:
    """Copy a deduplicated, trimmed slice of the alert database into a SQLite file.

    Kept: network alerts in the last WINDOW_HOURS of flow time (one per replayed flow, the
    lowest id), every alert a stored report refers to, all email alerts, stored reports,
    drift checks and retraining runs. Sampled flows and ground truth are left out.
    Explanations are kept where an alert has no shared campaign explanation to point to.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    dst = make_engine(f"sqlite:///{out}")
    init_db(dst)
    counts: dict[str, int] = {}
    with src.connect() as conn:
        latest = conn.execute(
            select(alerts.c.ts)
            .where(alerts.c.source == "network")
            .order_by(alerts.c.ts.desc())
            .limit(1)
        ).scalar()
        done = conn.execute(select(reports).where(reports.c.status == "done")).all()
        forced = {r.alert_id for r in done}
        start = latest - timedelta(hours=WINDOW_HOURS) if latest else None
        cond = alerts.c.source == "email"
        if start is not None:
            cond = cond | (alerts.c.ts >= start)
        cond = cond | alerts.c.id.in_(forced)
        seen: set[tuple[Any, ...]] = set()
        rows: list[dict[str, Any]] = []
        for r in conn.execute(select(alerts).where(cond).order_by(alerts.c.id)).yield_per(BATCH):
            d = dict(r._mapping)
            key = tuple(d[k] for k in DEDUP_KEY)
            if key in seen and d["id"] not in forced:
                continue
            seen.add(key)
            rows.append(d)
        kept = {d["id"] for d in rows}
        for d in rows:
            ev = d.get("evidence") or {}
            shared = ev.get("explained_by") if isinstance(ev, dict) else None
            if shared is not None and shared in kept and shared != d["id"]:
                d["reasons"] = None  # the page links to the explained alert
        with dst.begin() as w:
            for i in range(0, len(rows), BATCH):
                w.execute(insert(alerts), rows[i : i + BATCH])
            for table, data in (
                (reports, [dict(r._mapping) for r in done]),
                (drift_checks, [dict(r._mapping) for r in conn.execute(select(drift_checks))]),
                (retrain_runs, [dict(r._mapping) for r in conn.execute(select(retrain_runs))]),
            ):
                if data:
                    w.execute(insert(table), data)
                counts[table.name] = len(data)
        counts["alerts"] = len(rows)
    with dst.connect() as c:
        c.execute(text("VACUUM"))
    dst.dispose()
    counts["bytes"] = out.stat().st_size
    return counts


# -- guards --------------------------------------------------------------------------------


class RateLimiter:
    """At most `per_minute` requests per client in any 60 s window (in memory)."""

    def __init__(self, per_minute: int = RATE_PER_MINUTE) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = {}
        self.lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits.setdefault(client, deque())
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            if len(self.hits) > 10_000:  # forget idle clients
                self.hits = {k: v for k, v in self.hits.items() if v}
            return True


def client_of(request: Request) -> str:
    """The visitor's address: the first X-Forwarded-For hop behind the Spaces proxy."""
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


def guard(app: FastAPI, per_minute: int = RATE_PER_MINUTE) -> None:
    """Block writes and rate-limit scoring for a public deployment."""
    limiter = RateLimiter(per_minute)

    @app.middleware("http")
    async def demo_guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path, method = request.url.path, request.method
        if any(method == m and path.startswith(p) for m, p in BLOCKED):
            return JSONResponse({"detail": OFF}, status_code=403)
        if method == "POST" and path.startswith(LIMITED) and not limiter.allow(client_of(request)):
            return JSONResponse(
                {"detail": f"Demo limit: {per_minute} checks a minute. Try again shortly."},
                status_code=429,
            )
        return await call_next(request)
