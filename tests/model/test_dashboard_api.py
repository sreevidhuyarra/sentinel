"""Dashboard endpoints (overview, drift and retraining history, frontend routes) on SQLite."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert

from sentinel.db.schema import alerts, drift_checks, init_db, make_engine, retrain_runs
from sentinel.services import api as api_module
from sentinel.services.api import create_app

T0 = datetime(2017, 7, 7, 15, 0)


@pytest.fixture()
def client(tmp_path: Path) -> Any:
    engine = make_engine(f"sqlite:///{tmp_path / 'live.sqlite'}")
    init_db(engine)
    rows = [
        {
            "ts": T0 + timedelta(seconds=20 * i),
            "source": "network",
            "src_ip": src,
            "dst_ip": "192.168.10.50",
            "dst_port": 80,
            "predicted_family": fam,
            "severity": sev,
            "model_version": "t",
        }
        for i, (src, fam, sev) in enumerate(
            [("172.16.0.1", "DDoS", "Critical")] * 6 + [("192.168.10.8", "Bot", "High")] * 3
        )
    ]
    with engine.begin() as conn:
        conn.execute(insert(alerts), rows)
        conn.execute(
            insert(drift_checks),
            [
                {"ts": T0, "window": "all", "n": 500, "share": 0.05, "drifted": {}, "triggered": 0},
                {
                    "ts": T0 + timedelta(minutes=1),
                    "window": "all",
                    "n": 900,
                    "share": 0.31,
                    "drifted": {"syn_flag_count": 0.4},
                    "triggered": 1,
                },
            ],
        )
        conn.execute(
            insert(retrain_runs).values(
                started=T0,
                reason="drift",
                status="staged",
                n_extra=900,
                version="7",
                metrics={"candidate_value": 0.99},
            )
        )
    app = create_app(bundle=object(), db=engine)  # type: ignore[arg-type]
    with TestClient(app) as c:
        yield c


def test_overview_buckets_alerts_by_minute_and_family(client: TestClient) -> None:
    ov = client.get("/stats/overview", params={"minutes": 60}).json()
    assert ov["total"] == 9 and ov["by_family"] == {"DDoS": 6, "Bot": 3}
    assert ov["by_severity"] == {"Critical": 6, "High": 3}
    assert ov["top_sources"][0] == {"src_ip": "172.16.0.1", "alerts": 6}
    assert sum(sum(v for k, v in m.items() if k != "minute") for m in ov["per_minute"]) == 9


def test_drift_and_retrain_history(client: TestClient) -> None:
    d = client.get("/mlops/drift").json()
    assert [c["share"] for c in d["checks"]] == [0.05, 0.31]  # oldest first, for the chart
    assert d["checks"][1]["triggered"] == 1 and d["threshold"] > 0
    runs = client.get("/mlops/retrains").json()
    assert runs[0]["status"] == "staged" and runs[0]["metrics"]["candidate_value"] == 0.99
    assert "error" not in runs[0] and runs[0]["failed"] is False


def test_frontend_routes(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "FRONTEND_DIST", tmp_path / "missing")
    assert client.get("/app/").status_code == 404
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    monkeypatch.setattr(api_module, "FRONTEND_DIST", dist)
    assert client.get("/app/assets/app.js").text == "console.log(1)"
    assert "root" in client.get("/app/alerts/42").text  # client route -> index.html
    (tmp_path / "secret.txt").write_text("do not serve")  # outside dist
    leaked = client.get("/app/..%2Fsecret.txt")
    assert "do not serve" not in leaked.text and "root" in leaked.text  # never escapes dist
    assert client.get("/", follow_redirects=False).headers["location"] == "/app/"
