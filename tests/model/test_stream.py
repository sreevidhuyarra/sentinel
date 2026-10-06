"""Replayer -> detector end to end without Kafka: fakes, SQLite and the synthetic models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import func, select

from sentinel.anomaly.bundle import AnomalyBundle
from sentinel.db.schema import alerts, flows, init_db, make_engine
from sentinel.detection.fusion import Detector
from sentinel.ids.bundle import IDSBundle
from sentinel.stream import messages
from sentinel.stream.detector import StreamDetector, required_inputs, run
from sentinel.stream.replayer import replay, select_flows


class FakeProducer:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes, bytes | None]] = []

    def produce(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        self.sent.append((topic, value, key))

    def poll(self, timeout: float) -> int:
        return 0

    def flush(self, timeout: float) -> int:
        return 0


class FakeMessage:
    def __init__(self, value: bytes, ts_ms: int) -> None:
        self._value, self._ts = value, ts_ms

    def value(self) -> bytes:
        return self._value

    def error(self) -> None:
        return None

    def timestamp(self) -> tuple[int, int]:
        return (1, self._ts)


class FakeConsumer:
    def __init__(self, values: list[bytes]) -> None:
        self.pending = [FakeMessage(v, 1) for v in values]
        self.commits = 0

    def subscribe(self, topics: list[str]) -> None:
        self.topics = topics

    def consume(self, num_messages: int, timeout: float) -> list[FakeMessage]:
        out, self.pending = self.pending[:num_messages], self.pending[num_messages:]
        return out

    def commit(self, asynchronous: bool = False) -> None:
        self.commits += 1

    def assignment(self) -> list[Any]:
        return []


def test_replay_score_store_publish(trained_anomaly: dict[str, Any], tmp_path: Path) -> None:
    root, params = trained_anomaly["root"], trained_anomaly["params"]
    detector = Detector(IDSBundle.load(root / "bundle"), AnomalyBundle.load(root / "anomaly"))
    inputs = required_inputs(detector)

    frame = pl.read_parquet(params.resolve(params.data.processed_dir) / "flows.parquet")
    picked = select_flows(frame, "test", None, 400)
    assert picked["ts"].is_sorted()

    # Replayer: rate-limited, keyed by source, every model input in the message.
    out = FakeProducer()
    slept: list[float] = []
    clock = iter(range(10_000))
    n = replay(
        out,
        picked.iter_rows(named=True),
        inputs,
        "net.flows",
        rate=100,
        batch=100,
        clock=lambda: next(clock) * 0.01,
        sleep=slept.append,
    )
    assert n == len(picked) == len(out.sent) and slept  # it waited to hold 100 flows/s
    first = messages.decode(out.sent[0][1])
    assert set(first["flow"]) == set(inputs) and first["meta"]["src_ip"] == picked["src_ip"][0]
    assert out.sent[0][2] == str(picked["src_ip"][0]).encode()

    # Detector: micro-batches, alerts + flow sample in one transaction, alerts published.
    engine = make_engine(f"sqlite:///{tmp_path / 'live.sqlite'}")
    init_db(engine)
    published = FakeProducer()
    sd = StreamDetector(detector, engine, "test-model", published, sample_rate=0.5, seed=1)
    consumer = FakeConsumer([v for _, v, _ in out.sent])
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return not consumer.pending and calls["n"] > 1

    scored = run(consumer, sd, max_batch=150, max_wait=0.01, stop=stop)
    assert scored == len(picked) and consumer.commits == 3  # 150 + 150 + 100

    expected = sum(r["is_attack"] for r in detector.predict(picked.select(inputs).cast(pl.Float64)))
    with engine.connect() as conn:
        n_alerts = conn.execute(select(func.count()).select_from(alerts)).scalar_one()
        sample = conn.execute(select(flows.c.label, flows.c.features, flows.c.alert_id)).all()
    assert n_alerts == expected > 0
    assert 0.3 * len(picked) < len(sample) < 0.7 * len(picked)  # ~50% sampled
    assert all(set(json.loads(f) if isinstance(f, str) else f) == set(inputs) for _, f, _ in sample)
    assert any(lbl is not None for lbl, _, _ in sample)  # replay truth kept for retraining
    # One SHAP explanation per campaign; the rest share it and point at the explained alert.
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                alerts.c.id,
                alerts.c.reasons,
                alerts.c.evidence,
                alerts.c.src_ip,
                alerts.c.dst_ip,
                alerts.c.predicted_family,
            )
        ).all()
    by_id = {r[0]: r for r in rows}
    assert all(r[1] for r in rows)  # every alert has reasons
    pointers = [r for r in rows if r[2]]
    for r in pointers:
        ev = json.loads(r[2]) if isinstance(r[2], str) else r[2]
        rep = by_id[ev["explained_by"]]
        assert rep[3:] == r[3:] and not rep[2]  # same campaign; the representative has no pointer
    assert len(pointers) < len(rows)
    events = [json.loads(v) for t, v, _ in published.sent if t == "alerts.new"]
    assert len(events) == n_alerts and {"id", "predicted_family", "severity"} <= set(events[0])


def test_review_flag_stores_disagreements_as_low_alerts(
    trained_anomaly: dict[str, Any], tmp_path: Path
) -> None:
    from sentinel.detection.fusion import REVIEW

    root, params = trained_anomaly["root"], trained_anomaly["params"]
    ids = IDSBundle.load(root / "bundle")
    if ids.mlp is None or ids.lgbm_weight >= 1:
        return  # single-member ensemble: no disagreement signal to test
    detector = Detector(ids, AnomalyBundle.load(root / "anomaly"), review_delta=0.0)
    inputs = required_inputs(detector)
    frame = pl.read_parquet(params.resolve(params.data.processed_dir) / "flows.parquet")
    msgs = [
        messages.decode(messages.encode(r, inputs))
        for r in select_flows(frame, "test", None, 200).iter_rows(named=True)
    ]
    results = detector.predict(
        pl.DataFrame([m["flow"] for m in msgs], schema={c: pl.Float64 for c in inputs}),
        explain=False,
    )
    expected_review = sum(r["needs_review"] for r in results)
    assert expected_review > 0 and all(not r["is_attack"] for r in results if r["needs_review"])

    engine = make_engine(f"sqlite:///{tmp_path / 'live.sqlite'}")
    init_db(engine)
    out = StreamDetector(detector, engine, "m", explain=False).process(msgs)
    with engine.connect() as conn:
        review = conn.execute(
            select(alerts.c.severity, alerts.c.detector).where(alerts.c.predicted_family == REVIEW)
        ).all()
    assert (
        len(review) == expected_review
        and {r[0] for r in review} == {"Low"}
        and {r[1] for r in review} == {"review"}
    )
    assert out["alerts"] == expected_review + sum(r["is_attack"] for r in results)
    off = StreamDetector(detector, engine, "m", explain=False, review=False).process(msgs)
    assert off["alerts"] == sum(r["is_attack"] for r in results)
