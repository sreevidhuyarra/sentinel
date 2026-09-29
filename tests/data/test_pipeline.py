"""End-to-end data pipeline on the synthetic dataset, plus schema checks on the real one."""

import json
from pathlib import Path

import pandera.errors
import polars as pl
import pytest

from sentinel.common.config import Params, load_params
from sentinel.data.features import FeatureSpec
from sentinel.data.labels import FAMILIES
from sentinel.data.schemas import ProcessedFlowSchema, RawFlowSchema


def _processed(p: Params) -> pl.DataFrame:
    return pl.read_parquet(Path(p.data.processed_dir) / "flows.parquet")


def test_every_family_in_every_split(built: Params) -> None:
    df = _processed(built)
    for family in FAMILIES:
        assert set(df.filter(pl.col("family") == family)["split"]) == {"train", "val", "test"}, (
            family
        )


def test_processed_output_passes_schema(built: Params) -> None:
    ProcessedFlowSchema.validate(_processed(built), lazy=True)


def test_no_duplicates_or_nonfinite(built: Params) -> None:
    summary = json.loads((Path(built.data.reports_dir) / "data_summary.json").read_text())
    assert summary["clean"]["duplicates_dropped"] >= 25  # 5 injected per day
    assert summary["clean"]["non_finite_values"] > 0


def test_feature_spec_matches_processed(built: Params) -> None:
    spec = FeatureSpec.load(Path(built.data.processed_dir) / "feature_spec.json")
    x = spec.transform(_processed(built))
    assert x.width == len(spec.columns)
    assert "fwd_psh_flags" in spec.dropped_constant
    assert not set(spec.columns) & {"src_ip", "src_port", "row_id", "label", "family"}


def test_schema_rejects_bad_port(built: Params) -> None:
    df = pl.read_parquet(Path(built.data.interim_dir) / "flows.parquet")
    bad = df.with_columns(
        pl.when(pl.col("row_id") == 0).then(70000).otherwise(pl.col("dst_port")).alias("dst_port")
    )
    with pytest.raises(pandera.errors.SchemaErrors):
        RawFlowSchema.validate(bad, lazy=True)


@pytest.mark.slow
def test_real_processed_data_schema() -> None:
    p = load_params()
    path = p.resolve(p.data.processed_dir) / "flows.parquet"
    if not path.exists():
        pytest.skip("real dataset not built")
    ProcessedFlowSchema.validate(pl.read_parquet(path), lazy=True)
