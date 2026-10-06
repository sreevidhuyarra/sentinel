from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from sqlalchemy import func, insert, select

from sentinel.common.config import MlopsParams
from sentinel.db.schema import drift_checks, flows, init_db, make_engine
from sentinel.mlops.drift import compute_drift, live_window
from sentinel.mlops.watch import DriftWatcher

COLS = [f"f{i}" for i in range(10)]


def _frame(rng: np.random.Generator, n: int, shift: float = 0.0, shifted: int = 0) -> pl.DataFrame:
    data = {c: rng.normal(0, 1, n) + (shift if i < shifted else 0.0) for i, c in enumerate(COLS)}
    return pl.DataFrame(data)


def test_compute_drift_separates_same_from_shifted() -> None:
    rng = np.random.default_rng(0)
    ref = _frame(rng, 3000)
    same = compute_drift(_frame(rng, 1000), ref)
    moved = compute_drift(_frame(rng, 1000, shift=3.0, shifted=5), ref)
    assert same["share"] <= 0.1 and moved["share"] >= 0.5
    assert set(moved["drifted"]) >= {"f0", "f1", "f2", "f3", "f4"}
    assert list(moved["drifted"].values()) == sorted(moved["drifted"].values(), reverse=True)


def _store(engine: object, frame: pl.DataFrame, family: str) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    rows = [
        {"ts": now, "features": r, "predicted_family": family, "family": family, "scored_at": now}
        for r in frame.to_dicts()
    ]
    with engine.begin() as conn:  # type: ignore[attr-defined]
        conn.execute(insert(flows), rows)


def test_watcher_triggers_after_consecutive_checks_and_respects_cooldown(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    engine = make_engine(f"sqlite:///{tmp_path / 'live.sqlite'}")
    init_db(engine)
    started: list[str] = []
    clock = [1_000_000.0]
    cfg = MlopsParams(min_window_flows=100, consecutive=2, cooldown_minutes=30, drift_threshold=0.2)
    w = DriftWatcher(engine, _frame(rng, 2000), COLS, cfg, started.append, clock=lambda: clock[0])

    _store(engine, _frame(rng, 300), "Benign")
    calm = w.check()
    assert calm["all"]["share"] < 0.2 and not calm["triggered"]

    _store(engine, _frame(rng, 3000, shift=4.0, shifted=8), "DDoS")  # the mix changes
    assert live_window(engine, COLS, 10)["benign"].height == 300  # DDoS flows are not "benign"
    first = w.check()
    assert first["all"]["share"] >= 0.2 and not first["triggered"] and not started  # 1 of 2
    second = w.check()
    assert second["triggered"] and len(started) == 1 and "drift share" in started[0]
    clock[0] += 60  # still inside the cooldown
    w.check()
    w.check()
    assert len(started) == 1
    with engine.connect() as conn:
        n_checks = conn.execute(select(func.count()).select_from(drift_checks)).scalar_one()
        n_triggered = conn.execute(select(func.sum(drift_checks.c.triggered))).scalar_one()
    assert n_checks >= 8 and n_triggered == 1  # every window of every check is recorded


def test_retrain_pulls_only_labelled_known_families(tmp_path: Path) -> None:
    from sentinel.mlops.retrain import labelled_live_flows

    rng = np.random.default_rng(2)
    engine = make_engine(f"sqlite:///{tmp_path / 'live.sqlite'}")
    init_db(engine)
    _store(engine, _frame(rng, 5), "Benign")
    _store(engine, _frame(rng, 3), "PortScan")
    _store(engine, _frame(rng, 2), "NotAFamily")  # unknown label: ignored
    out = labelled_live_flows.fn(engine, COLS, limit=100)
    assert out.columns == [*COLS, "family"] and out.height == 8
    assert sorted(out["family"].unique().to_list()) == ["Benign", "PortScan"]
    newest = labelled_live_flows.fn(engine, COLS, limit=4)
    assert newest.height == 2 and set(newest["family"]) == {
        "PortScan"
    }  # newest first, unknown skipped
