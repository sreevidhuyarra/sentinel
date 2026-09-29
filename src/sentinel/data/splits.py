"""Leakage-aware train/val/test splits for flow data.

Random row splits leak: consecutive flows of one attack session are near-identical,
so the same session ends up on both sides. CIC-IDS2017 also runs each attack in a
single contiguous window on one day, so a plain "earlier days train, later days test"
split would leave whole classes unseen. The temporal split below therefore orders flows
in time *within each (day, label) group*, takes the earliest fraction for train, the
next for validation and the latest for test, and discards `gap_rows` at each boundary.
"""

from __future__ import annotations

import polars as pl

SPLIT_COL = "split"


def temporal_split(
    df: pl.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    gap_rows: int = 50,
    group_cols: tuple[str, ...] = ("day", "label"),
    order_col: str = "ts",
) -> pl.DataFrame:
    """Return `df` with a `split` column in {train, val, test}; gap rows are removed.

    The gap is capped at 2% of each group, so rare classes (Heartbleed has 11 flows,
    true Infiltration 36) lose almost nothing and still appear in every split.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("need 0 < train_frac, val_frac and train_frac + val_frac < 1")

    ordered = df.sort([*group_cols, order_col, "row_id"])
    rank = pl.int_range(pl.len()).over(group_cols)
    n = pl.len().over(group_cols)
    gap = pl.min_horizontal(pl.lit(gap_rows), n // 50)
    usable = n - 2 * gap
    train_end = (usable * train_frac).floor().clip(lower_bound=1)
    val_end = (usable * (train_frac + val_frac)).floor().clip(lower_bound=train_end + 1)

    split = (
        pl.when(rank < train_end)
        .then(pl.lit("train"))
        .when(rank < train_end + gap)
        .then(pl.lit(None, dtype=pl.String))
        .when(rank < val_end + gap)
        .then(pl.lit("val"))
        .when(rank < val_end + 2 * gap)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(pl.lit("test"))
    )
    return ordered.with_columns(split.alias(SPLIT_COL)).filter(pl.col(SPLIT_COL).is_not_null())


def holdout_family_split(
    df: pl.DataFrame, family: str, split_df: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Temporal split with one attack family removed from train and val entirely.

    Every flow of `family` goes to test so the anomaly detector can be scored on
    an attack type the supervised models never saw.
    """
    base = split_df if split_df is not None else temporal_split(df)
    return pl.concat(
        [
            base.filter(pl.col("family") != family),
            df.filter(pl.col("family") == family).with_columns(pl.lit("test").alias(SPLIT_COL)),
        ],
        how="diagonal_relaxed",
    )


def random_split(
    df: pl.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 42
) -> pl.DataFrame:
    """Naive stratified random split, kept only to show how much it inflates scores."""
    u = pl.int_range(pl.len()).shuffle(seed=seed).over("label") / pl.len().over("label")
    return df.with_columns(
        pl.when(u < train_frac)
        .then(pl.lit("train"))
        .when(u < train_frac + val_frac)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
        .alias(SPLIT_COL)
    )
