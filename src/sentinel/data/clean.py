"""Stage 2 cleaning: non-finite values, missing labels and duplicate flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl

from sentinel.data.columns import feature_columns


@dataclass
class CleanStats:
    rows_in: int
    non_finite_values: int
    null_labels_dropped: int
    duplicates_dropped: int
    rows_out: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def clean(df: pl.DataFrame) -> tuple[pl.DataFrame, CleanStats]:
    rows_in = df.height
    feats = feature_columns(df)
    floats = [c for c in feats if df.schema[c].is_float()]

    non_finite = (
        int(df.select(pl.sum_horizontal([(~pl.col(c).is_finite()).sum() for c in floats])).item())
        if floats
        else 0
    )
    # Rate features are undefined (x/0) for zero-duration flows; 0 is the value the
    # rate would have for a flow that moved nothing measurable per unit time.
    df = df.with_columns(
        [pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in floats]
    ).with_columns([pl.col(c).fill_null(0) for c in feats])

    before = df.height
    df = df.filter(pl.col("label").is_not_null() & (pl.col("label") != ""))
    null_labels = before - df.height

    before = df.height
    df = df.sort(["ts", "row_id"], nulls_last=True).unique(
        subset=[*feats, "label"], keep="first", maintain_order=True
    )
    dupes = before - df.height

    return df, CleanStats(rows_in, non_finite, null_labels, dupes, df.height)
