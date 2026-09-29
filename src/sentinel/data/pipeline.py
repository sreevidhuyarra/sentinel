"""Stage 2 'build': interim flows -> cleaned, labelled, split, validated Parquet + feature spec."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from sentinel.common.config import Params
from sentinel.common.logging import get_logger
from sentinel.data.clean import clean
from sentinel.data.features import FeatureSpec
from sentinel.data.labels import add_family_columns
from sentinel.data.schemas import ProcessedFlowSchema, RawFlowSchema
from sentinel.data.splits import temporal_split

log = get_logger(__name__)

REFERENCE_ROWS = 100_000
DEV_MIN_PER_LABEL = 200


def dev_sample(df: pl.DataFrame, frac: float, seed: int) -> pl.DataFrame:
    """Per-label sample that keeps rare labels whole (up to DEV_MIN_PER_LABEL rows)."""
    keep = pl.max_horizontal((pl.len() * frac).ceil(), pl.lit(DEV_MIN_PER_LABEL)).over(
        ["label", "split"]
    )
    rank = pl.int_range(pl.len()).shuffle(seed=seed).over(["label", "split"])
    return df.filter(rank < keep).sort("row_id")


def summarize(df: pl.DataFrame) -> dict[str, Any]:
    by_split = df.group_by(["split", "family"]).len().sort(["split", "family"])
    table: dict[str, dict[str, int]] = {}
    for split, family, n in by_split.iter_rows():
        table.setdefault(family, {})[split] = n
    return {"rows": df.height, "families": table}


def build(params: Params) -> dict[str, Any]:
    d, s, f = params.data, params.split, params.features
    interim = params.resolve(d.interim_dir) / "flows.parquet"
    processed_dir = params.resolve(d.processed_dir)
    reference_dir = params.resolve(d.reference_dir)

    df = pl.read_parquet(interim)
    RawFlowSchema.validate(df, lazy=True)

    df, stats = clean(df)
    df = add_family_columns(df, policy=d.attempted_policy)
    df = temporal_split(df, train_frac=s.train_frac, val_frac=s.val_frac, gap_rows=s.gap_rows)
    ProcessedFlowSchema.validate(df, lazy=True)

    processed_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(processed_dir / "flows.parquet", compression="zstd")
    dev_sample(df, d.dev_sample_frac, params.seed).write_parquet(
        processed_dir / "flows_dev.parquet", compression="zstd"
    )

    train = df.filter(pl.col("split") == "train")
    spec = FeatureSpec.fit(train, log_transform=f.log_transform, drop_constant=f.drop_constant)
    spec.save(processed_dir / "feature_spec.json")

    reference_dir.mkdir(parents=True, exist_ok=True)
    train.sample(min(REFERENCE_ROWS, train.height), seed=params.seed).write_parquet(
        reference_dir / "reference.parquet", compression="zstd"
    )

    summary = {
        "clean": stats.as_dict(),
        **summarize(df),
        "n_features": len(spec.columns),
        "dropped_constant": spec.dropped_constant,
    }
    reports = params.resolve(d.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("build done: %d rows, %d features", df.height, len(spec.columns))
    return summary
