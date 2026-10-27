"""Analyst feedback: verdicts on alerts and on checked emails, and the live precision they give.

An alert verdict also sets the alert's status and labels the alert's sampled flows that have
no label yet, so live traffic gains training labels (replayed flows keep their dataset label;
retraining already keeps live flows out of validation and test).
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import insert, select, update

from sentinel.db.schema import alerts, feedback, flows
from sentinel.ids.dataset import CLASSES

router = APIRouter()

ATTACK_FAMILIES = [c for c in CLASSES if c != "Benign"]


def _engine(request: Request) -> Any:
    from sentinel.services.api import _db_engine

    return _db_engine(request.app)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AlertFeedback(BaseModel):
    verdict: Literal["confirmed", "false_alarm"]
    family: str | None = None  # the right family, when the detector named the wrong one
    note: str = Field("", max_length=2000)


class EmailSnapshot(BaseModel):
    subject: str = Field("", max_length=2000)
    body: str = Field("", max_length=200_000)
    sender: str = Field("", max_length=1000)
    reply_to: str = Field("", max_length=1000)
    eml: str | None = Field(None, max_length=5_000_000)
    urls: list[str] = Field(default_factory=list, max_length=500)


class ResultSnapshot(BaseModel):
    verdict: Literal["malicious", "suspicious", "clean"]
    probability: float
    flagged_by: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)  # sender and intel checks that fired
    model_version: str | None = None
    url_model_version: str | None = None


class EmailFeedback(BaseModel):
    verdict: Literal["phishing", "legitimate"]
    email: EmailSnapshot
    result: ResultSnapshot
    note: str = Field("", max_length=2000)


@router.post("/feedback/alerts/{alert_id}")
def alert_feedback(alert_id: int, body: AlertFeedback, request: Request) -> dict[str, Any]:
    if body.family is not None and body.family not in ATTACK_FAMILIES:
        raise HTTPException(422, detail=f"family must be one of {ATTACK_FAMILIES}")
    if body.verdict == "false_alarm" and body.family:
        raise HTTPException(422, detail="a false alarm has no attack family")
    with _engine(request).begin() as conn:
        predicted = conn.execute(
            select(alerts.c.predicted_family).where(alerts.c.id == alert_id)
        ).scalar()
        if predicted is None:
            raise HTTPException(404, detail="alert not found")
        key = conn.execute(
            insert(feedback).values(
                created=_now(),
                kind="alert",
                alert_id=alert_id,
                verdict=body.verdict,
                family=body.family,
                note=body.note or None,
            )
        ).inserted_primary_key
        conn.execute(update(alerts).where(alerts.c.id == alert_id).values(status=body.verdict))
        family = "Benign" if body.verdict == "false_alarm" else body.family or predicted
        labelled = conn.execute(
            update(flows)
            .where(flows.c.alert_id == alert_id, flows.c.label.is_(None))
            .values(label=f"analyst: {body.verdict}", family=family)
        ).rowcount
    return {
        "feedback_id": int(key[0]) if key else None,
        "status": body.verdict,
        "family": family,
        "flows_labelled": int(labelled or 0),
    }


@router.get("/feedback/alerts/{alert_id}")
def alert_feedback_history(alert_id: int, request: Request) -> list[dict[str, Any]]:
    """Verdicts given on one alert, newest first."""
    with _engine(request).connect() as conn:
        rows = conn.execute(
            select(feedback.c.created, feedback.c.verdict, feedback.c.family, feedback.c.note)
            .where(feedback.c.kind == "alert", feedback.c.alert_id == alert_id)
            .order_by(feedback.c.id.desc())
        ).all()
    return [{**r._mapping, "created": r.created.replace(tzinfo=UTC).isoformat()} for r in rows]


@router.post("/feedback/email")
def email_feedback(body: EmailFeedback, request: Request) -> dict[str, Any]:
    with _engine(request).begin() as conn:
        key = conn.execute(
            insert(feedback).values(
                created=_now(),
                kind="email",
                verdict=body.verdict,
                note=body.note or None,
                model_verdict=body.result.verdict,
                email=body.email.model_dump(),
                result=body.result.model_dump(),
            )
        ).inserted_primary_key
    right = (body.verdict == "phishing") == (body.result.verdict != "clean")
    return {"feedback_id": int(key[0]) if key else None, "model_was_right": right}


def _share(num: int, den: int) -> float | None:
    return num / den if den else None


@router.get("/feedback/summary")
def feedback_summary(request: Request) -> dict[str, Any]:
    """Live precision from analyst verdicts (the latest verdict per alert counts).

    Only what analysts chose to label is counted, so these are not unbiased rates: they show
    how the detectors do on the alerts and emails people looked at.
    """
    with _engine(request).connect() as conn:
        alert_rows = conn.execute(
            select(feedback.c.alert_id, feedback.c.verdict, alerts.c.predicted_family)
            .join(alerts, alerts.c.id == feedback.c.alert_id)
            .where(feedback.c.kind == "alert")
            .order_by(feedback.c.id)
        ).all()
        email_rows = conn.execute(
            select(feedback.c.verdict, feedback.c.model_verdict).where(feedback.c.kind == "email")
        ).all()
    latest = {a: (v, fam) for a, v, fam in alert_rows}
    by_family: dict[str, Counter[str]] = {}
    for v, fam in latest.values():
        by_family.setdefault(fam, Counter())[v] += 1
    confirmed = sum(c["confirmed"] for c in by_family.values())
    false_alarms = sum(c["false_alarm"] for c in by_family.values())

    m: Counter[tuple[str, str]] = Counter((mv, v) for v, mv in email_rows)
    phishing = sum(n for (_, v), n in m.items() if v == "phishing")
    malicious = m[("malicious", "phishing")] + m[("malicious", "legitimate")]
    flagged_phish = m[("malicious", "phishing")] + m[("suspicious", "phishing")]
    return {
        "alerts": {
            "labelled": len(latest),
            "confirmed": confirmed,
            "false_alarms": false_alarms,
            "precision": _share(confirmed, confirmed + false_alarms),
            "by_family": {
                fam: {
                    "confirmed": c["confirmed"],
                    "false_alarms": c["false_alarm"],
                    "precision": _share(c["confirmed"], c["confirmed"] + c["false_alarm"]),
                }
                for fam, c in sorted(by_family.items())
            },
        },
        "emails": {
            "labelled": len(email_rows),
            # model verdict -> analyst verdict -> count
            "matrix": {
                mv: {v: m[(mv, v)] for v in ("phishing", "legitimate")}
                for mv in ("malicious", "suspicious", "clean")
            },
            "alert_precision": _share(m[("malicious", "phishing")], malicious),
            "flagged_share_of_phishing": _share(flagged_phish, phishing),
            "missed": m[("clean", "phishing")],
            "false_alarms": m[("malicious", "legitimate")],
        },
    }


def labelled_emails(engine: Any) -> list[dict[str, Any]]:
    """Every email verdict with what was checked: fresh test data for the email checks."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                feedback.c.id,
                feedback.c.created,
                feedback.c.verdict,
                feedback.c.model_verdict,
                feedback.c.email,
                feedback.c.result,
            )
            .where(feedback.c.kind == "email")
            .order_by(feedback.c.id)
        ).all()
    return [dict(r._mapping) for r in rows]
