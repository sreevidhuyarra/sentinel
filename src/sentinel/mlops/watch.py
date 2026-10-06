"""The drift job: check live drift every few minutes, export it, trigger retraining.

Retraining starts when the chosen window's drift share stays at or above the threshold for
`consecutive` checks, at most once per `cooldown_minutes`. It runs in a separate process
(`sentinel mlops retrain`), so a 15-minute training run never blocks the drift checks.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import polars as pl
from prometheus_client import Counter, Gauge
from sqlalchemy import Engine, insert, select

from sentinel.common.config import MlopsParams
from sentinel.common.logging import get_logger
from sentinel.db.schema import drift_checks, retrain_runs
from sentinel.mlops.drift import compute_drift, live_window

log = get_logger(__name__)

DRIFT_SHARE = Gauge(
    "sentinel_drift_share", "Share of features drifted vs the reference", ["window"]
)
DRIFT_N = Gauge("sentinel_drift_window_flows", "Flows in the drift window", ["window"])
DRIFT_PSI = Gauge("sentinel_drift_feature_psi", "Per-feature drift score (PSI)", ["feature"])
DRIFT_THRESHOLD = Gauge("sentinel_drift_threshold", "Drift share that triggers retraining")
DRIFT_DETECTED = Gauge("sentinel_drift_detected", "1 while drift is above the threshold")
RETRAINS = Counter(
    "sentinel_retrain_runs_total", "Retraining runs started by the drift job", ["outcome"]
)


class DriftWatcher:
    def __init__(
        self,
        engine: Engine,
        reference: pl.DataFrame,
        inputs: list[str],
        cfg: MlopsParams,
        start_retrain: Callable[[str], Any],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.engine, self.reference, self.inputs, self.cfg = engine, reference, inputs, cfg
        self.start_retrain, self.clock = start_retrain, clock
        self.above = 0
        self.last_retrain = self._last_retrain_time()
        DRIFT_THRESHOLD.set(cfg.drift_threshold)

    def _last_retrain_time(self) -> float:
        with self.engine.connect() as conn:
            last = conn.execute(
                select(retrain_runs.c.started).order_by(retrain_runs.c.id.desc())
            ).first()
        return last[0].replace(tzinfo=UTC).timestamp() if last else 0.0

    def check(self) -> dict[str, Any]:
        windows = live_window(self.engine, self.inputs, self.cfg.window_minutes)
        now = datetime.now(UTC).replace(tzinfo=None)
        out: dict[str, Any] = {}
        rows = []
        for name, frame in windows.items():
            if len(frame) < self.cfg.min_window_flows:
                out[name] = {"n": len(frame), "share": None}
                DRIFT_N.labels(name).set(len(frame))
                continue
            d = compute_drift(frame, self.reference, self.cfg.method, self.cfg.feature_threshold)
            out[name] = d
            DRIFT_SHARE.labels(name).set(d["share"])
            DRIFT_N.labels(name).set(d["n"])
            if name == self.cfg.trigger_window:
                DRIFT_PSI.clear()
                for feature, score in list(d["drifted"].items())[:10]:
                    DRIFT_PSI.labels(feature).set(score)
            rows.append(
                {
                    "ts": now,
                    "window": name,
                    "n": d["n"],
                    "share": d["share"],
                    "drifted": dict(list(d["drifted"].items())[:20]),
                    "triggered": 0,
                }
            )
        share = (out.get(self.cfg.trigger_window) or {}).get("share")
        detected = share is not None and share >= self.cfg.drift_threshold
        DRIFT_DETECTED.set(int(detected))
        self.above = self.above + 1 if detected else 0
        cooled = self.clock() - self.last_retrain >= 60 * self.cfg.cooldown_minutes
        trigger = detected and self.above >= self.cfg.consecutive and cooled
        if trigger:
            reason = (
                f"{self.cfg.trigger_window}-window drift share {share:.2f} >= "
                f"{self.cfg.drift_threshold:.2f} for {self.above} checks"
            )
            log.warning("drift detected: %s; starting retraining", reason)
            self.start_retrain(reason)
            RETRAINS.labels("started").inc()
            self.last_retrain, self.above = self.clock(), 0
            for r in rows:
                if r["window"] == self.cfg.trigger_window:
                    r["triggered"] = 1
        if rows:
            with self.engine.begin() as conn:
                conn.execute(insert(drift_checks), rows)
        out["triggered"] = trigger
        return out


def spawn_retrain(reason: str) -> subprocess.Popen[bytes]:
    """Run the Prefect retraining flow in its own process."""
    return subprocess.Popen(
        [sys.executable, "-m", "sentinel.cli", "mlops", "retrain", "--reason", reason]
    )
