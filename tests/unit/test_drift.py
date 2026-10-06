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


def _store(
    engine: object, frame: pl.DataFrame, family: str | None, predicted: str | None = None
) -> None:
    """Store sampled flows with ground truth `family` (None = unlabelled), scored as
    `predicted` (default: correctly)."""
    now = datetime.now(UTC).replace(tzinfo=None)
    rows = [
        {
            "ts": now,
            "features": r,
            "predicted_family": predicted or family,
            "family": family,
            "scored_at": now,
        }
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
    cfg = MlopsParams(
        min_window_flows=100,
        consecutive=2,
        cooldown_minutes=30,
        drift_threshold=0.2,
        trigger_window="all",
    )
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


def _watcher(tmp_path: Path, rng: np.random.Generator) -> tuple[DriftWatcher, list[str], object]:
    engine = make_engine(f"sqlite:///{tmp_path / 'live.sqlite'}")
    init_db(engine)
    started: list[str] = []
    cfg = MlopsParams(min_window_flows=100, consecutive=2, min_labelled_flows=100)
    return DriftWatcher(engine, _frame(rng, 2000), COLS, cfg, started.append), started, engine


def test_new_attack_mix_the_model_catches_does_not_retrain(tmp_path: Path) -> None:
    """The Module 6 demo case: the "all" window drifts, normal traffic and recall do not."""
    rng = np.random.default_rng(2)
    w, started, engine = _watcher(tmp_path, rng)
    _store(engine, _frame(rng, 400), "Benign")
    _store(engine, _frame(rng, 3000, shift=4.0, shifted=8), "DDoS")  # caught: predicted DDoS
    for _ in range(3):
        out = w.check()
    assert out["all"]["share"] >= 0.2 and out["benign"]["share"] < 0.2
    assert out["performance"]["attack_recall"] == 1.0 and out["performance"]["benign_fpr"] == 0
    assert not started
    with engine.connect() as conn:  # type: ignore[attr-defined]
        recalls = conn.execute(select(drift_checks.c.attack_recall)).scalars().all()
    assert recalls and all(r == 1.0 for r in recalls)


def test_missed_attacks_trigger_retraining(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    w, started, engine = _watcher(tmp_path, rng)
    _store(engine, _frame(rng, 400), "Benign")
    _store(engine, _frame(rng, 150), "Bot", predicted="Benign")  # slipped through
    _store(engine, _frame(rng, 150), "Bot")
    assert not w.check()["triggered"]  # 1 of 2 consecutive
    out = w.check()
    assert out["performance"]["attack_recall"] == 0.5
    assert out["triggered"] and "attack recall 0.500 < 0.900" in started[0]


def test_false_alarms_trigger_retraining_but_small_samples_do_not(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    w, started, engine = _watcher(tmp_path, rng)
    _store(engine, _frame(rng, 50), "Benign", predicted="PortScan")  # 50 labelled: too few
    _store(engine, _frame(rng, 250), None, predicted="DDoS")  # unlabelled: not counted
    w.check()
    w.check()
    assert not started
    _store(engine, _frame(rng, 200), "Benign")  # 250 benign, 20% false alarms
    w.check()
    out = w.check()
    assert out["performance"]["benign_fpr"] == 0.2
    assert out["triggered"] and "false-alarm rate 0.200 > 0.040" in started[0]


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


def test_runs_left_running_by_a_killed_process_are_closed(tmp_path: Path) -> None:
    from datetime import timedelta

    from sentinel.db.schema import retrain_runs
    from sentinel.mlops.retrain import close_stale_runs

    engine = make_engine(f"sqlite:///{tmp_path / 'live.sqlite'}")
    init_db(engine)
    now = datetime.now(UTC).replace(tzinfo=None)
    with engine.begin() as conn:
        conn.execute(
            insert(retrain_runs),
            [
                {"started": now - timedelta(hours=7), "reason": "old", "status": "running"},
                {"started": now - timedelta(minutes=5), "reason": "new", "status": "running"},
                {"started": now - timedelta(hours=9), "reason": "done", "status": "staged"},
            ],
        )
    assert close_stale_runs(engine, timeout_minutes=120) == 1
    with engine.connect() as conn:
        status = dict(conn.execute(select(retrain_runs.c.reason, retrain_runs.c.status)).all())
    assert status == {"old": "interrupted", "new": "running", "done": "staged"}
