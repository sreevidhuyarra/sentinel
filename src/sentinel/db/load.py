"""Fill the alerts table by scoring held-out data with the deployed detectors.

Until the streaming detector exists (Module 6), this is how alerts reach the database:
network alerts from the fused IDS on the CIC-IDS2017 test split, email alerts from the
phishing + URL models on a stratified sample of the email test split.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from sqlalchemy import Engine, delete, insert, select

from sentinel.common.logging import get_logger
from sentinel.db.schema import alert_truth, alerts, reports
from sentinel.detection.fusion import Detector
from sentinel.phishing.explain import explain
from sentinel.phishing.export import PhishingOnnxModel
from sentinel.phishing.urls import UrlModel

log = get_logger(__name__)

NO_DATE = datetime(2000, 1, 1)
META = ["ts", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label", "family"]
# Raw flow statistics kept with each alert, so the copilot can describe behaviour in words.
FEATURES = [
    "flow_duration",
    "total_fwd_packet",
    "total_bwd_packets",
    "total_length_of_fwd_packet",
    "total_length_of_bwd_packet",
    "flow_packets_s",
    "syn_flag_count",
    "rst_flag_count",
    "fin_flag_count",
    "psh_flag_count",
]


def _batches(df: pl.DataFrame, size: int) -> Iterator[pl.DataFrame]:
    for start in range(0, len(df), size):
        yield df.slice(start, size)


def clear(engine: Engine, source: str | None = None) -> None:
    with engine.begin() as conn:
        if source is None:
            conn.execute(delete(reports))
            conn.execute(delete(alert_truth))
            conn.execute(delete(alerts))
            return
        ids = [r[0] for r in conn.execute(select(alerts.c.id).where(alerts.c.source == source))]
        for chunk in range(0, len(ids), 10_000):
            part = ids[chunk : chunk + 10_000]
            conn.execute(delete(reports).where(reports.c.alert_id.in_(part)))
            conn.execute(delete(alert_truth).where(alert_truth.c.alert_id.in_(part)))
            conn.execute(delete(alerts).where(alerts.c.id.in_(part)))


def _insert(engine: Engine, rows: list[dict[str, Any]], truth: list[dict[str, Any]]) -> list[int]:
    if not rows:
        return []
    with engine.begin() as conn:
        res = conn.execute(
            insert(alerts).returning(alerts.c.id, sort_by_parameter_order=True), rows
        )
        ids = [int(r[0]) for r in res]
        conn.execute(
            insert(alert_truth),
            [{**t, "alert_id": i} for t, i in zip(truth, ids, strict=True)],
        )
    return ids


def load_network(
    engine: Engine,
    detector: Detector,
    flows: pl.DataFrame,
    model_version: str,
    batch_size: int = 20_000,
) -> int:
    """Score flows (already filtered to one split) and store every alert the detector raises."""
    n = 0
    for part in _batches(flows.sort("ts"), batch_size):
        results = detector.predict(part, explain=True)
        meta = part.select(META).to_dicts()
        feats = part.select([c for c in FEATURES if c in part.columns]).to_dicts()
        rows, truth = [], []
        for m, f, r in zip(meta, feats, results, strict=True):
            if not r["is_attack"]:
                continue
            rows.append(network_alert_row(m, f, r, model_version))
            truth.append({"label": m["label"], "family": m["family"]})
        n += len(_insert(engine, rows, truth))
        log.info("network alerts stored: %d", n)
    return n


def network_alert_row(
    meta: dict[str, Any], feats: dict[str, Any], result: dict[str, Any], model_version: str
) -> dict[str, Any]:
    """One `alerts` row from a scored flow (shared by the batch loader and the stream)."""
    return {
        "ts": meta["ts"],
        "source": "network",
        "src_ip": meta["src_ip"],
        "dst_ip": meta["dst_ip"],
        "src_port": meta["src_port"],
        "dst_port": meta["dst_port"],
        "protocol": meta["protocol"],
        "predicted_family": result["family"],
        "detector": result["detector"],
        "confidence": result["confidence"],
        "attack_score": result["attack_score"],
        "anomaly_score": result["anomaly_score"],
        "severity": result["severity"],
        "reasons": [
            {k: v for k, v in x.items() if k in ("feature", "value", "text", "contribution")}
            for x in result["reasons"]
        ],
        "evidence": None,
        "features": {
            k: None if feats.get(k) is None else float(feats[k]) for k in FEATURES if k in feats
        },
        "model_version": model_version,
    }


def load_email(
    engine: Engine,
    model: PhishingOnnxModel,
    url_model: UrlModel | None,
    emails: pl.DataFrame,
    per_kind: dict[str, int],
    seed: int,
    model_version: str,
) -> int:
    """Score a stratified email sample; store the flagged ones with their text as evidence."""
    rng = np.random.default_rng(seed)
    parts = []
    for kind, k in per_kind.items():
        sub = emails.filter(pl.col("kind") == kind)
        if len(sub):
            parts.append(
                sub[np.sort(rng.choice(len(sub), min(k, len(sub)), replace=False)).tolist()]
            )
    sample = pl.concat(parts)
    texts = sample["text"].to_list()
    prob = model.predict_proba(texts)
    rows, truth = [], []
    for i, rec in enumerate(sample.iter_rows(named=True)):
        urls = list(rec["urls"] or [])[:20]
        url_prob = url_model.predict_proba(urls) if url_model is not None and urls else np.array([])
        text_flag = bool(prob[i] >= model.threshold)
        if not text_flag:
            continue
        strong = prob[i] >= 0.999 or bool((url_prob >= 0.99).any())
        rows.append(
            {
                # Several corpora carry no date; place those at the load's epoch.
                "ts": rec["date"] if rec["date"] is not None else NO_DATE,
                "source": "email",
                "sender_domain": rec["sender_domain"],
                "predicted_family": "Phishing/Spam",
                "detector": "text",
                "confidence": float(prob[i]),
                "attack_score": float(prob[i]),
                "severity": "High" if strong else "Medium",
                "reasons": explain(model, texts[i]),
                "evidence": {
                    "email_text": texts[i],
                    "urls": [
                        {"url": u, "malicious_probability": float(p)}
                        for u, p in zip(urls, url_prob, strict=False)
                    ]
                    or [{"url": u, "malicious_probability": None} for u in urls],
                },
                "model_version": model_version,
            }
        )
        truth.append({"label": rec["kind"], "family": rec["kind"]})
        if len(rows) % 200 == 0:
            log.info("email alerts explained: %d", len(rows))
    return len(_insert(engine, rows, truth))
