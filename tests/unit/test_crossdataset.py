from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sentinel.data.features import FeatureSpec
from sentinel.data.synthetic import HEADER
from sentinel.ids.calibration import TemperatureBias
from sentinel.ids.crossdataset import (
    load_unsw,
    psi,
    recall_at_fpr,
    shared_spec,
    transfer_metrics,
)
from sentinel.ids.dataset import BENIGN, CLASSES

rng = np.random.default_rng(0)


def test_shared_spec_drops_missing_columns() -> None:
    spec = FeatureSpec(
        columns=["dst_port", "icmp_code", "total_tcp_flow_time"],
        log_columns=["total_tcp_flow_time"],
    )
    out = shared_spec(spec, {"dst_port", "flow_duration"})
    assert out.columns == ["dst_port"] and out.log_columns == []


def test_psi_zero_for_same_distribution_and_large_for_shift() -> None:
    a = rng.normal(0, 1, 20000)
    assert psi(a, rng.normal(0, 1, 20000)) < 0.01
    assert psi(a, rng.normal(3, 1, 20000)) > 1.0


def test_psi_handles_constant_feature() -> None:
    assert psi(np.zeros(100), np.ones(100)) > 1.0


def test_recall_at_fpr() -> None:
    is_attack = np.array([False] * 100 + [True] * 10)
    score = np.concatenate([np.linspace(0, 0.5, 100), np.full(10, 0.9)])
    recall, t = recall_at_fpr(is_attack, score, 0.01)
    assert recall == 1.0 and 0.49 < t < 0.9


def test_transfer_metrics_per_category() -> None:
    labels = pl.Series(["Benign"] * 4 + ["DoS"] * 2 + ["Exploits"] * 2)
    logits = np.zeros((8, len(CLASSES)))
    logits[:4, BENIGN] = 10
    logits[4:6, CLASSES.index("DoS")] = 10
    logits[6, CLASSES.index("WebAttack")] = 10
    logits[7, BENIGN] = 10  # missed
    proba = TemperatureBias.identity(len(CLASSES)).apply(logits)
    m = transfer_metrics(proba, 0.5, labels)
    assert m["benign_fpr"] == 0.0 and m["attack_recall"] == pytest.approx(0.75)
    assert m["per_category"]["DoS"]["flagged"] == 1.0
    assert m["per_category"]["DoS"]["as_DoS"] == 1.0
    assert m["per_category"]["Exploits"]["flagged"] == 0.5
    assert m["per_category"]["Exploits"]["predicted_as"] == {"Benign": 1, "WebAttack": 1}


def test_load_unsw_normalises_and_cleans(tmp_path: Path) -> None:
    cols = [
        c
        for c in HEADER
        if c not in ("id", "ICMP Code", "ICMP Type", "Total TCP Flow Time", "Attempted Category")
    ]
    n = 6
    data: dict[str, list[object]] = {c: [1] * n for c in cols}
    data["Timestamp"] = ["22/01/2015 07:50:15 AM"] * n
    data["Label"] = ["Benign", "Benign", "Exploits", "Exploits", "Fuzzers", "Benign"]
    data["Flow Bytes/s"] = [float("inf"), 1.0, 2.0, 3.0, 4.0, 5.0]
    data["Src Port"] = list(range(n))
    data["Flow Duration"] = [1, 1, 2, 3, 4, 5]
    csv = tmp_path / "flows.csv"
    pl.DataFrame(data).write_csv(csv)
    df = load_unsw(csv, cache=tmp_path / "cache.parquet")
    assert not {"icmp_code", "icmp_type", "total_tcp_flow_time"} & set(df.columns)
    assert {"dst_port", "flow_bytes_s", "label", "ts"} <= set(df.columns)
    assert df["flow_bytes_s"].is_finite().all()
    assert df["ts"].null_count() == 0
    assert (tmp_path / "cache.parquet").exists()
