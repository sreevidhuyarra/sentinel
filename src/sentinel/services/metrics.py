"""Prometheus metrics for the API and the copilot (report section 9: Metrics)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HTTP_SECONDS = Histogram(
    "sentinel_http_request_seconds",
    "API request latency by route template",
    ["route", "method", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
COPILOT_REPORTS = Counter(
    "sentinel_copilot_reports_total", "Incident reports produced", ["provider", "verified"]
)
COPILOT_SECONDS = Histogram(
    "sentinel_copilot_report_seconds",
    "Time to produce one incident report (whole graph)",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)
LLM_TOKENS = Counter(
    "sentinel_llm_tokens_total", "LLM tokens used by the copilot", ["provider", "direction"]
)
GUARD_FLAGS = Counter(
    "sentinel_guard_flags_total", "Untrusted fields in which the injection guard redacted text"
)


def instrument(app: FastAPI) -> None:
    """Latency per route template (bounded label set) and a /metrics endpoint."""

    @app.middleware("http")
    async def _timing(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        if path != "/metrics":
            HTTP_SECONDS.labels(path, request.method, str(response.status_code)).observe(
                time.perf_counter() - t0
            )
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_report(report: object, seconds: float) -> None:
    provider = getattr(report, "provider", None) or "none"
    verification = getattr(report, "verification", {}) or {}
    usage = getattr(report, "usage", {}) or {}
    COPILOT_REPORTS.labels(provider, str(bool(verification.get("passed")))).inc()
    COPILOT_SECONDS.observe(seconds)
    LLM_TOKENS.labels(provider, "input").inc(usage.get("input_tokens", 0))
    LLM_TOKENS.labels(provider, "output").inc(usage.get("output_tokens", 0))
    for g in getattr(report, "guard", []) or []:
        if g.get("flagged_segments"):
            GUARD_FLAGS.inc()
