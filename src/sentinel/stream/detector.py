"""The stream detector (report section 3.1): consume net.flows in micro-batches, score with
the fused IDS + anomaly detector, store alerts and a flow sample, publish alerts.new.

Offsets are committed only after the database write, so a crash replays a batch rather
than losing it (at-least-once).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
from sqlalchemy import Engine, bindparam, insert, update

from sentinel.common.logging import get_logger
from sentinel.db.load import network_alert_row
from sentinel.db.schema import alerts, flows
from sentinel.detection.fusion import REVIEW, Detector
from sentinel.stream import messages
from sentinel.stream.metrics import (
    ALERTS,
    BATCH_SECONDS,
    BATCH_SIZE,
    END_TO_END,
    FLOWS,
    LAG,
    SCORE_SECONDS,
)
from sentinel.stream.replayer import Producer

log = get_logger(__name__)


def required_inputs(detector: Detector) -> list[str]:
    cols = set(detector.ids.spec.input_columns)
    if detector.anomaly is not None:
        cols |= set(detector.anomaly.spec.input_columns)
    return sorted(cols)


@dataclass
class StreamDetector:
    detector: Detector
    engine: Engine
    model_version: str
    producer: Producer | None = None
    alert_topic: str = "alerts.new"
    sample_rate: float = 0.05
    explain: bool = True
    max_explain: int = 50  # campaigns explained per micro-batch
    review: bool = True  # store review-flag flows as low-severity alerts
    seed: int = 0
    inputs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.inputs = self.inputs or required_inputs(self.detector)
        self._rng = np.random.default_rng(self.seed)

    def _explain(
        self,
        df: pl.DataFrame,
        msgs: list[dict[str, Any]],
        results: list[dict[str, Any]],
        hits: list[int],
    ) -> dict[int, int]:
        """Explain one alert per campaign (source, target, family) in the batch.

        A port scan or flood puts hundreds of near-identical alerts in one batch; SHAP for
        every one of them cost ~1.1 s per 500 flows against 46 ms for scoring. Members share
        their representative's reasons (and point at it). At most `max_explain` campaigns
        per batch are explained, which bounds the worst case (a flood from many sources).
        Returns alert row -> representative row.
        """
        reps: dict[tuple[Any, ...], int] = {}
        member_of: dict[int, int] = {}
        for i in hits:
            key = campaign_key(msgs[i]["meta"], results[i])
            if key not in reps and len(reps) < self.max_explain:
                reps[key] = i
            if key in reps:
                member_of[i] = reps[key]
        rep_rows = sorted(set(reps.values()))
        if not rep_rows:
            return {}
        explained = self.detector.predict(df[rep_rows], explain=True)
        reasons = {i: e["reasons"] for i, e in zip(rep_rows, explained, strict=True)}
        for i, rep in member_of.items():
            results[i]["reasons"] = reasons[rep]
        return member_of

    def process(
        self, msgs: list[dict[str, Any]], sent_at: list[float] | None = None
    ) -> dict[str, Any]:
        """Score one micro-batch of decoded messages and store the outcome."""
        t0 = time.perf_counter()
        df = pl.DataFrame(
            [{c: m["flow"].get(c) for c in self.inputs} for m in msgs],
            schema={c: pl.Float64 for c in self.inputs},
        )
        t_score = time.perf_counter()
        results = self.detector.predict(df, explain=False)
        SCORE_SECONDS.observe(time.perf_counter() - t_score)
        now = datetime.now(UTC).replace(tzinfo=None)
        hits = [i for i, r in enumerate(results) if r["is_attack"]]
        if self.review:
            # Module 4 review flag: benign verdicts the ensemble's members disagree on.
            for i, r in enumerate(results):
                if r.get("needs_review"):
                    r.update(family=REVIEW, detector="review", severity="Low", confidence=None)
                    hits.append(i)
            hits.sort()
        explained_by = self._explain(df, msgs, results, hits) if self.explain else {}
        rows = [
            network_alert_row(msgs[i]["meta"], msgs[i]["flow"], results[i], self.model_version)
            for i in hits
        ]
        sampled = np.flatnonzero(self._rng.random(len(msgs)) < self.sample_rate)
        with self.engine.begin() as conn:
            ids: list[int] = []
            if rows:
                # Group members point at the alert whose explanation they share.
                rep_pos = {i: n for n, i in enumerate(hits)}
                for n, i in enumerate(hits):
                    rep = explained_by.get(i)
                    if rep is not None and rep != i:
                        rows[n]["evidence"] = {"explained_by_row": rep_pos[rep]}
                res = conn.execute(
                    insert(alerts).returning(alerts.c.id, sort_by_parameter_order=True), rows
                )
                ids = [int(r[0]) for r in res]
                shared = [
                    {"aid": ids[n], "ev": {"explained_by": ids[r["evidence"]["explained_by_row"]]}}
                    for n, r in enumerate(rows)
                    if r["evidence"]
                ]
                if shared:
                    conn.execute(
                        update(alerts)
                        .where(alerts.c.id == bindparam("aid"))
                        .values(evidence=bindparam("ev")),
                        shared,
                    )
            alert_of = dict(zip(hits, ids, strict=True))
            if len(sampled):
                conn.execute(
                    insert(flows),
                    [
                        {
                            "ts": msgs[i]["meta"]["ts"],
                            "src_ip": msgs[i]["meta"].get("src_ip"),
                            "dst_ip": msgs[i]["meta"].get("dst_ip"),
                            "dst_port": msgs[i]["meta"].get("dst_port"),
                            "features": {c: msgs[i]["flow"].get(c) for c in self.inputs},
                            "predicted_family": results[i]["family"],
                            "label": msgs[i].get("truth", {}).get("label"),
                            "family": msgs[i].get("truth", {}).get("family"),
                            "alert_id": alert_of.get(int(i)),
                            "model_version": self.model_version,
                            "scored_at": now,
                        }
                        for i in sampled
                    ],
                )
        if self.producer is not None:
            for aid, row in zip(ids, rows, strict=True):
                event = {
                    "id": aid,
                    "ts": row["ts"].isoformat() if isinstance(row["ts"], datetime) else row["ts"],
                    **{
                        k: row[k]
                        for k in (
                            "src_ip",
                            "dst_ip",
                            "dst_port",
                            "predicted_family",
                            "severity",
                            "confidence",
                            "detector",
                        )
                    },
                }
                self.producer.produce(self.alert_topic, json.dumps(event, default=str).encode())
            self.producer.poll(0)
        seconds = time.perf_counter() - t0
        FLOWS.inc(len(msgs))
        BATCH_SECONDS.observe(seconds)
        BATCH_SIZE.observe(len(msgs))
        for row in rows:
            ALERTS.labels(row["predicted_family"], row["severity"] or "none").inc()
        if sent_at:
            done = time.time()
            for t in sent_at:
                END_TO_END.observe(max(0.0, done - t))
        return {"flows": len(msgs), "alerts": len(ids), "sampled": len(sampled), "seconds": seconds}


def campaign_key(meta: dict[str, Any], result: dict[str, Any]) -> tuple[Any, ...]:
    return (meta.get("src_ip"), meta.get("dst_ip"), result["family"])


def run(
    consumer: Any,
    sd: StreamDetector,
    topic: str = "net.flows",
    max_batch: int = 500,
    max_wait: float = 0.2,
    stop: Callable[[], bool] = lambda: False,
) -> int:
    """Consume until `stop()`; returns the number of flows scored."""
    consumer.subscribe([topic])
    total, last_lag = 0, 0.0
    while not stop():
        batch = consumer.consume(num_messages=max_batch, timeout=max_wait)
        good = [m for m in batch if m.error() is None]
        for m in batch:
            if m.error() is not None:
                log.warning("consume error: %s", m.error())
        if not good:
            continue
        decoded = [messages.decode(m.value()) for m in good]
        sent = [m.timestamp()[1] / 1000 for m in good if m.timestamp()[1] > 0]
        out = sd.process(decoded, sent)
        consumer.commit(asynchronous=False)
        total += out["flows"]
        if time.monotonic() - last_lag > 5:
            LAG.set(_lag(consumer))
            last_lag = time.monotonic()
            log.info(
                "scored %d flows; last batch %d flows, %d alerts, %.0f ms",
                total,
                out["flows"],
                out["alerts"],
                1000 * out["seconds"],
            )
    return total


def _lag(consumer: Any) -> float:
    lag = 0
    for tp in consumer.assignment():
        _, high = consumer.get_watermark_offsets(tp, timeout=1, cached=True)
        pos = consumer.position([tp])[0].offset
        if high >= 0 and pos >= 0:
            lag += high - pos
    return float(lag)
