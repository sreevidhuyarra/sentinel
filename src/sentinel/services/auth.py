"""Optional access control: HTTP Basic auth on the API, the dashboard and the alert feed.

Off by default: a laptop SOC that listens on 127.0.0.1. Setting SENTINEL_API_KEY turns it
on: any user name, the key as the password. Browsers ask once and resend the credentials on
same-origin requests (including the WebSocket handshake), so the dashboard needs no login
page. /health and /metrics stay open for container health checks and Prometheus.
"""

from __future__ import annotations

import base64
import binascii
import secrets
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

OPEN_PATHS = frozenset({"/health", "/metrics"})
CHALLENGE = {"WWW-Authenticate": 'Basic realm="Sentinel", charset="UTF-8"'}


def authorized(header: str | None, key: str) -> bool:
    """True if an Authorization header carries Basic credentials whose password is `key`."""
    if not header or header[:6].lower() != "basic ":
        return False
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return False
    _, _, password = decoded.partition(":")
    return secrets.compare_digest(password.encode(), key.encode())


def protect(app: FastAPI, key: str) -> None:
    """Require the key on every HTTP route except OPEN_PATHS (WebSockets check it themselves)."""

    @app.middleware("http")
    async def require_key(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in OPEN_PATHS or authorized(request.headers.get("authorization"), key):
            return await call_next(request)
        return Response("Authentication required", status_code=401, headers=CHALLENGE)
