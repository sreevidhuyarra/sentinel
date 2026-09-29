"""End-to-end Module 2 on synthetic data: training, studies, registry, fusion and API."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl
from fastapi.testclient import TestClient

from sentinel.anomaly.bundle import AnomalyBundle
from sentinel.detection.fusion import UNKNOWN, Detector
from sentinel.ids.bundle import IDSBundle
from sentinel.ids.dataset import load_splits
from sentinel.services.api import create_app


def _bundles(t: dict[str, Any]) -> tuple[IDSBundle, AnomalyBundle]:
    return IDSBundle.load(t["root"] / "bundle"), AnomalyBundle.load(t["root"] / "anomaly")


def test_detectors_beat_chance_and_respect_budget(trained_anomaly: dict[str, Any]) -> None:
    out = trained_anomaly["out"]
    assert out["bottleneck"] in (4, 8) and len(out["bottleneck_sweep"]) == 2
    for name in ("autoencoder", "iforest"):
        t = out[name]["test"]
        assert t["roc_auc"] > 0.6, name
        assert t["benign_fpr"] <= 0.05, name


def test_holdout_study_fused_never_worse(trained_anomaly: dict[str, Any]) -> None:
    holdout = trained_anomaly["out"]["holdout"]
    assert set(holdout) == {"Infiltration", "Bot", "WebAttack"}
    for fam, h in holdout.items():
        assert h["flows"] > 0, fam
        assert h["fused_recall"] >= max(h["supervised_recall"], h["autoencoder_recall"]), fam


def test_fusion_study_and_registry(trained_anomaly: dict[str, Any]) -> None:
    out = trained_anomaly["out"]
    f = out["fusion"]
    assert f["fused"]["attack_recall"] >= f["supervised"]["attack_recall"]
    assert out["registry"]["model"] == "anomaly-detector" and out["registry"]["promoted"]
    for name in ("results.md", "model_card.md", "metrics.json", "figures/autoencoder_scores.png"):
        assert (trained_anomaly["root"] / "anomaly_reports" / name).exists(), name


def test_bundle_roundtrip_and_reasons(trained_anomaly: dict[str, Any]) -> None:
    _, an = _bundles(trained_anomaly)
    frame = load_splits(trained_anomaly["params"]).test.frame.head(50)
    again = AnomalyBundle.load(trained_anomaly["root"] / "anomaly")
    assert np.allclose(an.score(frame), again.score(frame))
    reasons = an.reasons(frame.head(3))
    assert len(reasons) == 3 and all(len(r) == 5 for r in reasons)
    assert "unusual for normal traffic" in reasons[0][0]["text"]


def test_detector_turns_benign_anomalies_into_unknown(trained_anomaly: dict[str, Any]) -> None:
    ids, an = _bundles(trained_anomaly)
    frame = load_splits(trained_anomaly["params"]).test.frame
    benign = frame.filter(pl.col("family") == "Benign").head(30)
    # Threshold 0 makes every flow anomalous: every flow the IDS calls benign must become
    # an Unknown anomaly alert, and supervised alerts must keep their family.
    det = Detector(ids, replace(an, threshold=0.0))
    sup = ids.predict(benign)
    fused = det.predict(benign)
    for s_row, f_row in zip(sup, fused, strict=True):
        if s_row["is_attack"]:
            assert f_row["family"] == s_row["family"] and f_row["detector"] == "supervised"
        else:
            assert f_row["family"] == UNKNOWN and f_row["detector"] == "anomaly"
            assert f_row["confidence"] is None and f_row["severity"] in ("Low", "Medium")
            assert len(f_row["reasons"]) == 5
    # With the real threshold nearly all benign flows stay benign.
    real = Detector(ids, an).predict(benign)
    assert sum(r["family"] == UNKNOWN for r in real) <= 3


def test_api_with_anomaly_model(trained_anomaly: dict[str, Any]) -> None:
    ids, an = _bundles(trained_anomaly)
    app = create_app(ids, source="ids-test", anomaly=an, anomaly_source="anomaly-test")
    flows = load_splits(trained_anomaly["params"]).test.frame.head(5)
    with TestClient(app) as c:
        assert c.get("/health").json()["anomaly_model_loaded"] is True
        assert c.get("/models").json()["anomaly"]["source"] == "anomaly-test"
        body = c.post(
            "/score/flows", json={"flows": flows.select(ids.spec.input_columns).to_dicts()}
        ).json()
    assert body["anomaly_model_version"] == "anomaly-test"
    assert all(r["anomaly_score"] is not None for r in body["results"])
