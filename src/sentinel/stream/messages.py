"""The net.flows message: model inputs, connection metadata and (replay only) ground truth.

`truth` stands in for labels that would arrive later from analysts on a live network. The
detector never scores with it; it is only stored with sampled flows for retraining and for
measuring live accuracy.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

META_FIELDS = ("ts", "src_ip", "dst_ip", "src_port", "dst_port", "protocol")


def encode(row: dict[str, Any], inputs: list[str]) -> bytes:
    meta = {k: row.get(k) for k in META_FIELDS}
    if isinstance(meta["ts"], datetime):
        meta["ts"] = meta["ts"].isoformat()
    msg = {
        "meta": meta,
        "flow": {c: row.get(c) for c in inputs},
        "truth": {"label": row.get("label"), "family": row.get("family")},
    }
    return json.dumps(msg, separators=(",", ":"), default=str).encode()


def decode(raw: bytes) -> dict[str, Any]:
    msg: dict[str, Any] = json.loads(raw)
    ts = msg["meta"].get("ts")
    if isinstance(ts, str):
        msg["meta"]["ts"] = datetime.fromisoformat(ts)
    return msg
