import math
from pathlib import Path

import polars as pl

from sentinel.data.clean import clean
from sentinel.data.features import FeatureSpec


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "ts": [None] * 4,
            "label": ["BENIGN", "BENIGN", "DDoS", "DDoS"],
            "dst_port": [443, 443, 80, 80],
            "flow_duration": [0, 0, 5_000_000, 10],
            "total_fwd_packet": [1, 1, 3, 2],
            "total_bwd_packets": [0, 0, 2, 0],
            "total_length_of_fwd_packet": [0, 0, 9000, 10],
            "total_length_of_bwd_packet": [0, 0, 100, 0],
            "flow_bytes_s": [math.inf, math.inf, 1800.0, float("nan")],
            "always_zero": [0, 0, 0, 0],
        },
        schema_overrides={"ts": pl.Datetime("us")},
    )


def test_clean_fixes_non_finite_and_duplicates() -> None:
    out, stats = clean(_df())
    assert stats.non_finite_values == 3
    assert stats.duplicates_dropped == 1  # rows 0 and 1 are identical
    assert out["flow_bytes_s"].is_finite().all()
    assert out.height == 3


def test_feature_spec_fit_transform_roundtrip(tmp_path: Path) -> None:
    train, _ = clean(_df())
    spec = FeatureSpec.fit(train)
    assert "always_zero" in spec.dropped_constant
    assert "dst_port" in spec.columns and "dst_port" not in spec.log_columns
    assert "flow_duration" in spec.log_columns
    assert {"fwd_bwd_bytes_ratio", "fwd_bwd_packets_ratio"} <= set(spec.columns)

    x = spec.transform(train)
    assert x.columns == spec.columns
    assert all(t == pl.Float32 for t in x.dtypes)
    assert x["flow_duration"].max() < 20  # log1p(5e6) ~ 15.4

    spec.save(tmp_path / "spec.json")
    assert FeatureSpec.load(tmp_path / "spec.json") == spec
