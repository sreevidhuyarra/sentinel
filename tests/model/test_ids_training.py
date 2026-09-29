"""End-to-end IDS training on synthetic data with a throwaway MLflow (SQLite) registry."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient

from sentinel.ids.bundle import IDSBundle
from sentinel.ids.dataset import load_splits
from sentinel.services.api import create_app


def test_minimum_quality_on_synthetic(trained: dict[str, Any]) -> None:
    models = trained["out"]["models"]
    assert set(models) == {"logreg", "lightgbm", "xgboost", "mlp", "ensemble"}
    # Synthetic families overlap (several share port 80 with random packet stats), so
    # the bar is a sanity floor, not a quality target.
    assert models["ensemble"]["test"]["macro_f1"] >= 0.75
    assert models["lightgbm"]["test"]["macro_f1"] >= models["logreg"]["test"]["macro_f1"] - 0.02
    assert models["ensemble"]["test"]["benign_fpr"] <= 0.02


def test_registered_and_promoted(trained: dict[str, Any]) -> None:
    reg = trained["out"]["registry"]
    assert reg["version"] == "1" and reg["promoted"] is True


def test_reports_written(trained: dict[str, Any]) -> None:
    reports = trained["root"] / "reports"
    for name in (
        "results.md",
        "model_card.md",
        "metrics.json",
        "figures/confusion_ensemble.png",
        "figures/shap_dos.png",
    ):
        assert (reports / name).exists(), name


def test_bundle_roundtrip_is_exact(trained: dict[str, Any]) -> None:
    b = IDSBundle.load(trained["root"] / "bundle")
    s = load_splits(trained["params"])
    again = IDSBundle.load(trained["root"] / "bundle")
    assert np.allclose(b.proba(s.test.X[:200]), again.proba(s.test.X[:200]))
    assert np.allclose(b.proba(s.test.X[:200]).sum(axis=1), 1)


def test_prediction_invariant_to_identifiers(trained: dict[str, Any]) -> None:
    b = IDSBundle.load(trained["root"] / "bundle")
    frame = load_splits(trained["params"]).test.frame.head(300)
    moved = frame.with_columns(
        pl.lit("10.9.9.9").alias("src_ip"), pl.lit(1).alias("src_port"), pl.lit(0).alias("row_id")
    )
    assert [r["family"] for r in b.predict(frame)] == [r["family"] for r in b.predict(moved)]


def test_shap_contributions_sum_to_logits(trained: dict[str, Any]) -> None:
    b = IDSBundle.load(trained["root"] / "bundle")
    X = load_splits(trained["params"]).test.X[:50]
    assert np.allclose(b.lgbm.contributions(X).sum(axis=2), b.lgbm.logits(X), atol=1e-6)


def test_alerts_carry_reasons(trained: dict[str, Any]) -> None:
    b = IDSBundle.load(trained["root"] / "bundle")
    test = load_splits(trained["params"]).test.frame
    out = b.predict(test.filter(pl.col("family") == "BruteForce").head(20))
    alerts = [r for r in out if r["is_attack"]]
    assert alerts and all(len(r["reasons"]) == 5 and r["severity"] for r in alerts)


@pytest.fixture(scope="module")
def client(trained: dict[str, Any]) -> Iterator[TestClient]:
    app = create_app(IDSBundle.load(trained["root"] / "bundle"), source="test-bundle")
    with TestClient(app) as c:
        yield c


def _flows(trained: dict[str, Any], n: int = 5) -> list[dict[str, Any]]:
    b = IDSBundle.load(trained["root"] / "bundle")
    frame = load_splits(trained["params"]).test.frame.head(n)
    return frame.select(b.spec.input_columns).to_dicts()


def test_api_health_and_models(client: TestClient) -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "model_loaded": True,
        "anomaly_model_loaded": False,
    }
    assert client.get("/models").json()["ids"]["source"] == "test-bundle"


def test_api_scores_flows(client: TestClient, trained: dict[str, Any]) -> None:
    r = client.post("/score/flows", json={"flows": _flows(trained)})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 5
    assert all(abs(sum(x["probabilities"].values()) - 1) < 1e-6 for x in body["results"])


def test_api_rejects_missing_features(client: TestClient) -> None:
    r = client.post("/score/flows", json={"flows": [{"dst_port": 22}]})
    assert r.status_code == 422
    assert "flow_duration" in r.json()["detail"]["missing_features"]


def test_api_rejects_empty_batch(client: TestClient) -> None:
    assert client.post("/score/flows", json={"flows": []}).status_code == 422
