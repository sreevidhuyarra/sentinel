"""Endpoints behind the analyst dashboard (report section 10): live stats, model health,
related alerts and the robustness lab."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from sentinel.common.config import PROJECT_ROOT, get_settings
from sentinel.copilot.tools import AlertTools
from sentinel.db.schema import alerts, drift_checks, flows, retrain_runs

router = APIRouter()

ADVERSARIAL_RESULTS = PROJECT_ROOT / "reports" / "adversarial" / "results.json"
FAMILIES = ["DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration"]


def _engine(request: Request) -> Any:
    from sentinel.services.api import _db_engine

    return _db_engine(request.app)


def _since(minutes: int) -> datetime:
    return datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=minutes)


@router.get("/stats/overview")
def overview(request: Request, minutes: int = Query(60, ge=1, le=24 * 60)) -> dict[str, Any]:
    """Alert volume for the Live Overview page, by when the detector stored the alerts.

    Stream alerts carry the replayed flow time in `ts`; the live view needs wall-clock
    time, which is the scoring time of the sampled flows and the insert order of alerts.
    Here alerts are bucketed by their flow timestamp within the window covered by the
    most recent alerts, so a replay of 2017 traffic still shows its timeline.
    """
    engine = _engine(request)
    with engine.connect() as conn:
        latest = conn.execute(
            select(func.max(alerts.c.ts)).where(alerts.c.source == "network")
        ).scalar()
        if latest is None:
            return {
                "window_minutes": minutes,
                "total": 0,
                "per_minute": [],
                "by_severity": {},
                "by_family": {},
                "top_sources": [],
                "latest": None,
            }
        start = latest - timedelta(minutes=minutes)
        cond = (alerts.c.ts >= start) & (alerts.c.source == "network")
        rows = conn.execute(select(alerts.c.ts, alerts.c.predicted_family).where(cond)).all()
        sev = conn.execute(
            select(alerts.c.severity, func.count()).where(cond).group_by(alerts.c.severity)
        ).all()
        top = conn.execute(
            select(alerts.c.src_ip, func.count().label("n"))
            .where(cond)
            .group_by(alerts.c.src_ip)
            .order_by(func.count().desc())
            .limit(5)
        ).all()
    per_minute: dict[str, dict[str, int]] = {}
    by_family: dict[str, int] = {}
    for ts, fam in rows:
        key = ts.replace(second=0, microsecond=0).isoformat()
        per_minute.setdefault(key, {})
        per_minute[key][fam] = per_minute[key].get(fam, 0) + 1
        by_family[fam] = by_family.get(fam, 0) + 1
    return {
        "window_minutes": minutes,
        "latest": latest.isoformat(),
        "total": len(rows),
        "per_minute": [{"minute": k, **v} for k, v in sorted(per_minute.items())],
        "by_severity": {s or "none": n for s, n in sev},
        "by_family": dict(sorted(by_family.items(), key=lambda kv: -kv[1])),
        "top_sources": [{"src_ip": s, "alerts": n} for s, n in top],
    }


@router.get("/stats/throughput")
def throughput(minutes: int = Query(30, ge=1, le=24 * 60)) -> dict[str, Any]:
    """Flows scored per second over time, from Prometheus (empty if it is not reachable)."""
    end = time.time()
    query: dict[str, str | float] = {
        "query": "sum(rate(sentinel_flows_scored_total[1m]))",
        "start": end - 60 * minutes,
        "end": end,
        "step": max(15, minutes * 2),
    }
    try:
        r = httpx.get(
            f"{get_settings().prometheus_url}/api/v1/query_range", params=query, timeout=5
        )
        r.raise_for_status()
        result = r.json()["data"]["result"]
    except (httpx.HTTPError, KeyError, ValueError):
        return {"available": False, "points": []}
    values = result[0]["values"] if result else []
    return {
        "available": True,
        "points": [
            {"t": datetime.fromtimestamp(float(t), UTC).isoformat(), "flows_per_s": float(v)}
            for t, v in values
        ],
    }


@router.get("/alerts/{alert_id}/related")
def related(alert_id: int, request: Request) -> dict[str, Any]:
    tools = AlertTools(_engine(request))
    alert = tools.get_alert(alert_id)
    if alert is None:
        raise HTTPException(404, detail="alert not found")
    return tools.related_alerts(alert)


@router.get("/mlops/drift")
def drift_history(request: Request, limit: int = Query(200, ge=1, le=2000)) -> dict[str, Any]:
    from sentinel.common.config import load_params

    with _engine(request).connect() as conn:
        rows = conn.execute(
            select(drift_checks).order_by(drift_checks.c.id.desc()).limit(limit)
        ).all()
        n_flows = conn.execute(select(func.count()).select_from(flows)).scalar_one()
    p = load_params()
    return {
        "threshold": p.mlops.drift_threshold,
        "trigger_window": p.mlops.trigger_window,
        "sampled_flows": n_flows,
        "checks": [dict(r._mapping) for r in reversed(rows)],
    }


@router.get("/mlops/retrains")
def retrain_history(request: Request, limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
    with _engine(request).connect() as conn:
        rows = conn.execute(
            select(retrain_runs).order_by(retrain_runs.c.id.desc()).limit(limit)
        ).all()
    return [
        {k: v for k, v in r._mapping.items() if k != "error"} | {"failed": bool(r.error)}
        for r in rows
    ]


@router.get("/robustness/study")
def robustness_study() -> dict[str, Any]:
    """The Module 4 study (reports/adversarial/results.json), trimmed for the dashboard."""
    if not ADVERSARIAL_RESULTS.exists():
        raise HTTPException(404, detail="run the Module 4 study first (dvc repro -s robustness)")
    r = json.loads(ADVERSARIAL_RESULTS.read_text())
    return {
        k: r[k]
        for k in (
            "targets",
            "clean",
            "review_flag",
            "feature_space",
            "hop_skip_jump",
            "problem_space",
            "problem_space_by_family",
        )
        if k in r
    }


class RobustnessRun(BaseModel):
    pad: float = Field(0.5, ge=0, le=2, description="extra payload per forward packet (fraction)")
    delay: float = Field(3.0, ge=1, le=20, description="timing stretch factor")
    family: str | None = Field(None, description="one attack family, or all")
    n_samples: int = Field(300, ge=10, le=2000)


@router.post("/robustness/run")
def robustness_run(body: RobustnessRun, request: Request) -> dict[str, Any]:
    """Live problem-space attack (Module 4): pad and delay caught attack flows, re-score them.

    Uses test-split attack flows the deployed detector catches, applies the realistic
    padding/delay transform (dependent features recomputed) and reports how many now pass.
    """
    from sentinel.adversarial.threat import pad_and_delay
    from sentinel.common.config import load_params
    from sentinel.stream.detector import required_inputs

    detector = request.app.state.detector
    p = load_params()
    frame = pl.scan_parquet(p.resolve(p.data.processed_dir) / "flows.parquet").filter(
        (pl.col("split") == "test") & (pl.col("family") != "Benign")
    )
    if body.family:
        if body.family not in FAMILIES:
            raise HTTPException(422, detail=f"family must be one of {FAMILIES}")
        frame = frame.filter(pl.col("family") == body.family)
    sample = frame.collect()
    if sample.is_empty():
        raise HTTPException(404, detail="no test flows for that family")
    rng = np.random.default_rng(p.seed)
    idx = rng.choice(len(sample), min(body.n_samples, len(sample)), replace=False)
    sample = sample[np.sort(idx).tolist()]
    inputs = required_inputs(detector)
    before = detector.predict(sample.select(inputs).cast(pl.Float64), explain=False)
    caught = np.array([r["is_attack"] for r in before])
    attacked = pad_and_delay(sample.filter(pl.Series(caught)), body.pad, body.delay)
    after = detector.predict(attacked.select(inputs).cast(pl.Float64), explain=False)
    evaded = np.array([not r["is_attack"] for r in after])
    fams = sample.filter(pl.Series(caught))["family"].to_list()
    by_family: dict[str, list[int]] = {}
    for f, e in zip(fams, evaded, strict=True):
        by_family.setdefault(f, [0, 0])
        by_family[f][0] += int(e)
        by_family[f][1] += 1
    return {
        "pad": body.pad,
        "delay": body.delay,
        "sampled": len(sample),
        "caught_before": int(caught.sum()),
        "evaded_after": int(evaded.sum()),
        "evasion_rate": float(evaded.mean()) if len(evaded) else None,
        "by_family": {
            f: {"evaded": e, "caught_before": n, "rate": e / n}
            for f, (e, n) in sorted(by_family.items())
        },
        "note": "Detector = deployed supervised ensemble + anomaly fusion, no review flag.",
    }
