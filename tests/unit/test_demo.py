"""Public demo: snapshot slicing, blocked writes, stored-report-only investigations, rate limit."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, insert, select

from sentinel.common.config import get_settings
from sentinel.db.schema import alerts, init_db, make_engine
from sentinel.demo import RateLimiter, snapshot
from sentinel.services.api import create_app

T0 = datetime(2017, 7, 7, 15, 0)


def _alert(i: int, ts: datetime, **kw: object) -> dict[str, object]:
    return {
        "ts": ts,
        "source": "network",
        "src_ip": "172.16.0.1",
        "src_port": 1000 + i,
        "dst_ip": "192.168.10.50",
        "dst_port": 80,
        "protocol": 6,
        "predicted_family": "DDoS",
        "reasons": [{"text": "big packets", "contribution": 1.0}],
        "evidence": None,
        **kw,
    }


@pytest.fixture()
def source(tmp_path: Path) -> Engine:
    engine = make_engine(f"sqlite:///{tmp_path / 'src.sqlite'}")
    init_db(engine)
    rows = [_alert(i, T0 + timedelta(seconds=i)) for i in range(3)]  # ids 1-3
    rows += [_alert(i, T0 + timedelta(seconds=i)) for i in range(3)]  # ids 4-6: a replay
    rows.append(_alert(9, T0 - timedelta(hours=6)))  # id 7: outside the window
    rows.append(_alert(1, T0 + timedelta(seconds=1), evidence={"explained_by": 1}))  # id 8
    with engine.begin() as conn:
        conn.execute(insert(alerts), rows)
    return engine


def test_snapshot_keeps_one_alert_per_replayed_flow(source: Engine, tmp_path: Path) -> None:
    out = tmp_path / "demo.sqlite"
    counts = snapshot(source, out)
    assert counts["alerts"] == 3 and counts["bytes"] > 0
    with make_engine(f"sqlite:///{out}").connect() as conn:
        assert [r.id for r in conn.execute(select(alerts.c.id).order_by(alerts.c.id))] == [1, 2, 3]


def test_snapshot_strips_explanations_shared_with_a_kept_alert(tmp_path: Path) -> None:
    engine = make_engine(f"sqlite:///{tmp_path / 'src.sqlite'}")
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(
            insert(alerts),
            [_alert(0, T0), _alert(1, T0, src_port=2000, evidence={"explained_by": 1})],
        )
    out = tmp_path / "demo.sqlite"
    snapshot(engine, out)
    with make_engine(f"sqlite:///{out}").connect() as conn:
        reasons = dict(conn.execute(select(alerts.c.id, alerts.c.reasons)).all())
    assert reasons[1] and reasons[2] is None


def test_rate_limiter_window() -> None:
    lim = RateLimiter(per_minute=2)
    assert lim.allow("a", 0) and lim.allow("a", 1) and not lim.allow("a", 2)
    assert lim.allow("b", 2)  # per client
    assert lim.allow("a", 60.5)  # the first hit has left the window


@pytest.fixture()
def demo_client(source: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    get_settings.cache_clear()
    try:
        with TestClient(create_app(bundle=object(), db=source)) as c:  # type: ignore[arg-type]
            yield c
    finally:
        get_settings.cache_clear()


def test_demo_blocks_writes_and_live_investigations(demo_client: TestClient) -> None:
    assert demo_client.get("/health").json()["demo"] is True
    r = demo_client.post("/feedback/alerts/1", json={"verdict": "confirmed"})
    assert r.status_code == 403 and "public demo" in r.json()["detail"]
    assert demo_client.post("/robustness/run", json={}).status_code == 403
    r = demo_client.post("/copilot/investigate/1")
    assert r.status_code == 403 and "turned off in the public demo" in r.json()["detail"]
    assert demo_client.get("/alerts/1").status_code == 200  # reading still works


def test_demo_rate_limits_scoring_per_visitor(demo_client: TestClient) -> None:
    def post(ip: str) -> int:
        return demo_client.post(
            "/score/phishing", json={"body": "hi"}, headers={"X-Forwarded-For": ip}
        ).status_code

    codes = [post("203.0.113.7") for _ in range(21)]
    assert codes[:20] == [503] * 20  # no phishing model loaded here, but not limited yet
    assert codes[20] == 429
    assert post("203.0.113.9") == 503  # another visitor
