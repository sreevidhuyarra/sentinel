"""The copilot's database tools (report section 8.1, gather_context).

Every query is built from SQLAlchemy expressions over whitelisted columns of the `alerts`
table; there is no free-form SQL. The engine connects with a SELECT-only role (Postgres)
or a read-only file handle (SQLite), so even a bug here cannot write, and `alert_truth`
(evaluation labels) is outside the role's grants.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Engine, and_, func, or_, select

from sentinel.db.schema import alerts

COLUMNS = [
    "id",
    "ts",
    "source",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "sender_domain",
    "predicted_family",
    "detector",
    "confidence",
    "attack_score",
    "anomaly_score",
    "severity",
    "reasons",
    "evidence",
    "model_version",
    "status",
]
SUMMARY_COLUMNS = [c for c in COLUMNS if c not in ("reasons", "evidence")]
FILTERS = {
    "predicted_family",
    "severity",
    "src_ip",
    "dst_ip",
    "dst_port",
    "source",
    "sender_domain",
}
MAX_ROWS = 100


def _row(r: Any, columns: list[str]) -> dict[str, Any]:
    d = dict(zip(columns, r, strict=True))
    if isinstance(d.get("ts"), datetime):
        d["ts"] = d["ts"].isoformat(sep=" ", timespec="seconds")
    return d


class AlertTools:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def get_alert(self, alert_id: int) -> dict[str, Any] | None:
        cols = [alerts.c[c] for c in COLUMNS]
        with self.engine.connect() as conn:
            r = conn.execute(select(*cols).where(alerts.c.id == int(alert_id))).first()
        return None if r is None else _row(r, COLUMNS)

    def query_alerts(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 20,
        exclude_id: int | None = None,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Parameterised filter over whitelisted columns; at most MAX_ROWS rows."""
        unknown = set(filters) - FILTERS
        if unknown:
            raise ValueError(f"filter not allowed: {sorted(unknown)}")
        cond = [alerts.c[k] == v for k, v in filters.items() if v is not None]
        if since is not None:
            cond.append(alerts.c.ts >= since)
        if until is not None:
            cond.append(alerts.c.ts <= until)
        if exclude_id is not None:
            cond.append(alerts.c.id != exclude_id)
        q = (
            select(*[alerts.c[c] for c in SUMMARY_COLUMNS])
            .where(and_(*cond))
            .order_by(alerts.c.ts, alerts.c.id)
            .limit(max(1, min(int(limit), MAX_ROWS)))
        )
        with self.engine.connect() as conn:
            return [_row(r, SUMMARY_COLUMNS) for r in conn.execute(q)]

    def related_alerts(
        self, alert: dict[str, Any], window_minutes: int = 30, sample: int = 15
    ) -> dict[str, Any]:
        """Activity around the alert: same source (or sender domain) and same target."""
        ts = datetime.fromisoformat(alert["ts"])
        if alert["source"] == "email":
            window = timedelta(days=30)
            key = alerts.c.sender_domain == alert["sender_domain"]
        else:
            window = timedelta(minutes=window_minutes)
            key = or_(alerts.c.src_ip == alert["src_ip"], alerts.c.dst_ip == alert["dst_ip"])
        cond = and_(
            key,
            alerts.c.ts >= ts - window,
            alerts.c.ts <= ts + window,
            alerts.c.id != alert["id"],
        )
        agg = (
            select(
                alerts.c.predicted_family,
                alerts.c.src_ip,
                alerts.c.dst_ip,
                alerts.c.dst_port,
                func.count().label("n"),
                func.min(alerts.c.ts).label("first"),
                func.max(alerts.c.ts).label("last"),
            )
            .where(cond)
            .group_by(
                alerts.c.predicted_family, alerts.c.src_ip, alerts.c.dst_ip, alerts.c.dst_port
            )
            # Fixed order, so the same alert always yields the same prompt (cacheable,
            # reproducible): SQL leaves the order of groups and of ties undefined.
            .order_by(
                alerts.c.predicted_family, alerts.c.src_ip, alerts.c.dst_ip, alerts.c.dst_port
            )
        )
        with self.engine.connect() as conn:
            groups = conn.execute(agg).all()
            rows = conn.execute(
                select(*[alerts.c[c] for c in SUMMARY_COLUMNS])
                .where(cond)
                .order_by(func.abs(alerts.c.id - alert["id"]), alerts.c.id)
                .limit(sample)
            ).all()
        families: Counter[str] = Counter()
        ports: Counter[int] = Counter()
        sources: Counter[str] = Counter()
        targets: Counter[str] = Counter()
        first = last = None
        for fam, src, dst, port, n, lo, hi in groups:
            families[fam] += n
            if port is not None:
                ports[int(port)] += n
            if src:
                sources[src] += n
            if dst:
                targets[dst] += n
            first = lo if first is None or lo < first else first
            last = hi if last is None or hi > last else last
        return {
            "window": f"+/- {window}",
            "count": sum(families.values()),
            "families": dict(families.most_common()),
            "distinct_dst_ports": len(ports),
            "top_dst_ports": dict(ports.most_common(10)),
            "distinct_sources": len(sources),
            "top_sources": dict(sources.most_common(5)),
            "distinct_targets": len(targets),
            "top_targets": dict(targets.most_common(5)),
            "first": None if first is None else str(first),
            "last": None if last is None else str(last),
            "sample": sorted((_row(r, SUMMARY_COLUMNS) for r in rows), key=lambda d: d["ts"]),
        }
