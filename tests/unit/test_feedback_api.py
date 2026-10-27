"""Analyst feedback endpoints on SQLite: alert verdicts, flow labels, email verdicts, summary."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, insert, select

from sentinel.db.schema import alerts, flows, init_db, make_engine
from sentinel.services.api import create_app

T0 = datetime(2017, 7, 7, 15, 0)


@pytest.fixture()
def engine(tmp_path: Path) -> Engine:
    engine = make_engine(f"sqlite:///{tmp_path / 'feedback.sqlite'}")
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(
            insert(alerts),
            [
                {"ts": T0, "source": "network", "predicted_family": fam, "src_ip": "10.0.0.1"}
                for fam in ("DoS", "DoS", "PortScan")
            ],
        )
        conn.execute(
            insert(flows),
            [
                # Alert 1's flow is live (no label yet); alert 2's is replayed (dataset label).
                {
                    "ts": T0,
                    "features": {},
                    "alert_id": 1,
                    "scored_at": T0,
                    "label": None,
                    "family": None,
                },
                {
                    "ts": T0,
                    "features": {},
                    "alert_id": 2,
                    "scored_at": T0,
                    "label": "DoS Hulk",
                    "family": "DoS",
                },
            ],
        )
    return engine


@pytest.fixture()
def client(engine: Engine) -> Iterator[TestClient]:
    with TestClient(create_app(bundle=object(), db=engine)) as c:  # type: ignore[arg-type]
        yield c


def test_alert_verdict_sets_status_and_labels_only_unlabelled_flows(
    client: TestClient, engine: Engine
) -> None:
    r = client.post("/feedback/alerts/1", json={"verdict": "false_alarm", "note": "backup job"})
    assert r.status_code == 200 and r.json()["flows_labelled"] == 1
    assert r.json()["family"] == "Benign"
    r = client.post("/feedback/alerts/2", json={"verdict": "false_alarm"})
    assert r.json()["flows_labelled"] == 0  # the dataset label stays
    with engine.connect() as conn:
        status = dict(conn.execute(select(alerts.c.id, alerts.c.status)).all())
        labels = dict(conn.execute(select(flows.c.alert_id, flows.c.family)).all())
    assert status == {1: "false_alarm", 2: "false_alarm", 3: "new"}
    assert labels == {1: "Benign", 2: "DoS"}
    [h] = client.get("/feedback/alerts/1").json()
    assert h["verdict"] == "false_alarm" and h["note"] == "backup job"


def test_alert_verdict_validation(client: TestClient) -> None:
    assert client.post("/feedback/alerts/99", json={"verdict": "confirmed"}).status_code == 404
    bad_family = {"verdict": "confirmed", "family": "Martians"}
    assert client.post("/feedback/alerts/1", json=bad_family).status_code == 422
    benign = {"verdict": "false_alarm", "family": "DoS"}
    assert client.post("/feedback/alerts/1", json=benign).status_code == 422
    assert client.post("/feedback/alerts/1", json={"verdict": "maybe"}).status_code == 422


def test_summary_counts_the_latest_verdict_per_alert(client: TestClient) -> None:
    client.post("/feedback/alerts/1", json={"verdict": "false_alarm"})
    client.post("/feedback/alerts/1", json={"verdict": "confirmed"})  # changed mind
    client.post("/feedback/alerts/2", json={"verdict": "confirmed"})
    client.post("/feedback/alerts/3", json={"verdict": "false_alarm"})
    email = {"subject": "Invoice", "body": "Please pay", "sender": "a <b@c.com>"}
    for verdict, model in (
        ("phishing", "malicious"),
        ("legitimate", "malicious"),
        ("phishing", "clean"),
    ):
        r = client.post(
            "/feedback/email",
            json={
                "verdict": verdict,
                "email": email,
                "result": {"verdict": model, "probability": 0.5},
            },
        )
        assert r.status_code == 200
    assert r.json()["model_was_right"] is False
    s = client.get("/feedback/summary").json()
    a = s["alerts"]
    assert (a["labelled"], a["confirmed"], a["false_alarms"]) == (3, 2, 1)
    assert (
        a["by_family"]["DoS"]["precision"] == 1.0 and a["by_family"]["PortScan"]["precision"] == 0
    )
    e = s["emails"]
    assert e["labelled"] == 3 and e["alert_precision"] == 0.5
    assert e["missed"] == 1 and e["false_alarms"] == 1
    assert e["flagged_share_of_phishing"] == 0.5
