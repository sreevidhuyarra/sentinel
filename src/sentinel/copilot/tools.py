"""The copilot's database tools (report section 8.1, gather_context).

Every query is built from SQLAlchemy expressions over whitelisted columns of the `alerts`
table; there is no free-form SQL. The engine connects with a SELECT-only role (Postgres)
or a read-only file handle (SQLite), so even a bug here cannot write, and `alert_truth`
(evaluation labels) is outside the role's grants.
"""

from __future__ import annotations

import statistics
from collections import Counter
from datetime import datetime, timedelta
from itertools import pairwise
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
    "features",
    "model_version",
    "status",
]
SUMMARY_COLUMNS = [c for c in COLUMNS if c not in ("reasons", "evidence", "features")]
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

    def activity(self, alert: dict[str, Any], window_minutes: int = 30) -> dict[str, Any]:
        """Fan-out of the source, fan-in to the target and timing of the source->target pair.

        The raw material for behaviour descriptions: how many ports / hosts one source
        touched, how many sources hit one target, and whether the pair's connections are
        regular (beaconing) and alike (automated). Network alerts only.
        """
        if alert["source"] != "network" or not alert.get("src_ip"):
            return {}
        ts = datetime.fromisoformat(alert["ts"])
        win = and_(
            alerts.c.ts >= ts - timedelta(minutes=window_minutes),
            alerts.c.ts <= ts + timedelta(minutes=window_minutes),
        )
        src, dst = alerts.c.src_ip == alert["src_ip"], alerts.c.dst_ip == alert["dst_ip"]
        with self.engine.connect() as conn:
            fan_out = conn.execute(
                select(
                    func.count(),
                    func.count(func.distinct(alerts.c.dst_ip)),
                    func.count(func.distinct(alerts.c.dst_port)),
                ).where(win, src)
            ).one()
            fan_in = conn.execute(
                select(func.count(), func.count(func.distinct(alerts.c.src_ip))).where(win, dst)
            ).one()
            pair = conn.execute(
                select(alerts.c.ts, alerts.c.dst_port, alerts.c.features)
                .where(win, src, dst)
                .order_by(alerts.c.ts, alerts.c.id)
                .limit(2000)
            ).all()
        times = [r[0] for r in pair]
        gaps = [(b - a).total_seconds() for a, b in pairwise(times)]
        feats = [r[2] or {} for r in pair]

        def stat(key: str) -> dict[str, float] | None:
            vals = [float(f[key]) for f in feats if f.get(key) is not None]
            if not vals:
                return None
            mean = statistics.fmean(vals)
            sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            return {"mean": mean, "cv": sd / mean if mean else 0.0}

        gap_mean = statistics.fmean(gaps) if gaps else None
        return {
            "window_minutes": window_minutes,
            "from_source": {"alerts": fan_out[0], "hosts": fan_out[1], "ports": fan_out[2]},
            "to_target": {"alerts": fan_in[0], "sources": fan_in[1]},
            "pair": {
                "alerts": len(pair),
                "ports": len({r[1] for r in pair}),
                "interval_median_s": statistics.median(gaps) if gaps else None,
                "interval_cv": (statistics.pstdev(gaps) / gap_mean) if gaps and gap_mean else None,
                "fwd_bytes": stat("total_length_of_fwd_packet"),
                "bwd_bytes": stat("total_length_of_bwd_packet"),
                "duration_us": stat("flow_duration"),
                "fwd_packets": stat("total_fwd_packet"),
                "packets_per_s": stat("flow_packets_s"),
            },
            "flow": alert.get("features") or {},
        }
