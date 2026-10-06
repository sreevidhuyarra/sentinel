"""The drift job: check live drift and performance every minute, export both, retrain.

A check is "degraded" when the trigger window's drift share reaches the threshold (normal
traffic changed; by default the benign window), or when the labelled live flows show attack
recall below its floor or false alarms above their budget. Retraining starts after
`consecutive` degraded checks, at most once per `cooldown_minutes`, in a separate process
(`sentinel mlops retrain`) so a 15-minute training run never blocks the checks.
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
from sentinel.mlops.drift import compute_drift, live_performance, live_window

log = get_logger(__name__)

DRIFT_SHARE = Gauge(
    "sentinel_drift_share", "Share of features drifted vs the reference", ["window"]
)
DRIFT_N = Gauge("sentinel_drift_window_flows", "Flows in the drift window", ["window"])
DRIFT_PSI = Gauge("sentinel_drift_feature_psi", "Per-feature drift score (PSI)", ["feature"])
DRIFT_THRESHOLD = Gauge("sentinel_drift_threshold", "Drift share that triggers retraining")
DRIFT_DETECTED = Gauge("sentinel_drift_detected", "1 while drift is above the threshold")
LIVE_RECALL = Gauge("sentinel_live_attack_recall", "Alerted share of labelled live attacks")
LIVE_FPR = Gauge("sentinel_live_benign_fpr", "Alerted share of labelled live benign flows")
DEGRADED = Gauge("sentinel_model_degraded", "1 while a retraining condition holds")
RECALL_FLOOR = Gauge("sentinel_live_recall_floor", "Live attack recall that triggers retraining")
FPR_BUDGET = Gauge("sentinel_live_fpr_budget", "Live false-alarm rate that triggers retraining")
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
        from sentinel.mlops.retrain import close_stale_runs

        if n := close_stale_runs(engine, cfg.retrain_timeout_minutes):
            log.warning("marked %d retraining run(s) left running by a killed process", n)
        self.last_retrain = self._last_retrain_time()
        DRIFT_THRESHOLD.set(cfg.drift_threshold)
        RECALL_FLOOR.set(cfg.min_attack_recall)
        FPR_BUDGET.set(cfg.max_benign_fpr)

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
        perf = live_performance(self.engine, self.cfg.window_minutes)
        out["performance"] = perf
        for r in rows:
            r["attack_recall"], r["benign_fpr"] = perf["attack_recall"], perf["benign_fpr"]
        reasons = self._reasons(out, perf)
        degraded = bool(reasons)
        DEGRADED.set(int(degraded))
        self.above = self.above + 1 if degraded else 0
        cooled = self.clock() - self.last_retrain >= 60 * self.cfg.cooldown_minutes
        trigger = degraded and self.above >= self.cfg.consecutive and cooled
        if trigger:
            reason = f"{'; '.join(reasons)} for {self.above} checks"
            log.warning("model degraded: %s; starting retraining", reason)
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

    def _reasons(self, out: dict[str, Any], perf: dict[str, Any]) -> list[str]:
        cfg, reasons = self.cfg, []
        share = (out.get(cfg.trigger_window) or {}).get("share")
        drifted = share is not None and share >= cfg.drift_threshold
        DRIFT_DETECTED.set(int(drifted))
        if drifted:
            reasons.append(
                f"{cfg.trigger_window}-window drift share {share:.2f} >= {cfg.drift_threshold:.2f}"
            )
        recall, fpr = perf["attack_recall"], perf["benign_fpr"]
        if recall is not None:
            LIVE_RECALL.set(recall)
            if perf["n_attack"] >= cfg.min_labelled_flows and recall < cfg.min_attack_recall:
                reasons.append(f"live attack recall {recall:.3f} < {cfg.min_attack_recall:.3f}")
        if fpr is not None:
            LIVE_FPR.set(fpr)
            if perf["n_benign"] >= cfg.min_labelled_flows and fpr > cfg.max_benign_fpr:
                reasons.append(f"live false-alarm rate {fpr:.3f} > {cfg.max_benign_fpr:.3f}")
        return reasons


def spawn_retrain(reason: str) -> subprocess.Popen[bytes]:
    """Run the Prefect retraining flow in its own process."""
    return subprocess.Popen(
        [sys.executable, "-m", "sentinel.cli", "mlops", "retrain", "--reason", reason]
    )
