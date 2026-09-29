"""Pandera schemas enforced at ingest, after cleaning, and in CI on the synthetic sample."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

from sentinel.data.columns import CORE_FEATURES
from sentinel.data.labels import FAMILIES

MAX_NULL_RATE = 0.01
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
_NON_NEGATIVE = {
    "total_fwd_packet",
    "total_bwd_packets",
    "total_length_of_fwd_packet",
    "total_length_of_bwd_packet",
    "fwd_packet_length_max",
    "bwd_packet_length_max",
    "syn_flag_count",
    "ack_flag_count",
}


def _null_rate_ok(data: pa.PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(
        pl.all_horizontal(
            [(pl.col(c).null_count() / pl.len()) <= MAX_NULL_RATE for c in CORE_FEATURES]
        )
    )


def _finite(data: pa.PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(data.key).cast(pl.Float64).is_finite())


def _core_columns(finite: bool) -> dict[str, pa.Column]:
    cols: dict[str, pa.Column] = {}
    for c in CORE_FEATURES:
        checks = []
        if c == "dst_port":
            checks.append(pa.Check.in_range(0, 65535))
        elif c in _NON_NEGATIVE:
            checks.append(pa.Check.ge(0))
        if finite:
            checks.append(pa.Check(_finite, name="finite"))
        cols[c] = pa.Column(checks=checks, nullable=not finite)
    return cols


RawFlowSchema = pa.DataFrameSchema(
    {
        "row_id": pa.Column(pl.Int64, unique=True),
        "day": pa.Column(pl.String, pa.Check.isin(WEEKDAYS)),
        "label": pa.Column(pl.String, nullable=False),
        **_core_columns(finite=False),
    },
    checks=[pa.Check(_null_rate_ok, error=f"core feature null rate > {MAX_NULL_RATE:.0%}")],
    strict=False,
    name="raw_flows",
)

ProcessedFlowSchema = pa.DataFrameSchema(
    {
        "row_id": pa.Column(pl.Int64, unique=True),
        "day": pa.Column(pl.String, pa.Check.isin(WEEKDAYS)),
        "label": pa.Column(pl.String, nullable=False),
        "family": pa.Column(pl.String, pa.Check.isin(FAMILIES)),
        "attempted": pa.Column(pl.Boolean),
        "split": pa.Column(pl.String, pa.Check.isin(["train", "val", "test"])),
        **_core_columns(finite=True),
    },
    strict=False,
    name="processed_flows",
)
