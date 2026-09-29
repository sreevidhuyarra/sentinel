"""Stage 1: raw CICFlowMeter CSVs -> one canonical Parquet file (data/interim/flows.parquet)."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from sentinel.common.logging import get_logger
from sentinel.data.columns import ID_COLUMNS, LABEL_COLUMNS, normalize_columns

log = get_logger(__name__)

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_TS_FORMATS = ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"]
_TEXT_COLUMNS = frozenset(ID_COLUMNS + LABEL_COLUMNS)


def day_from_filename(path: Path) -> str:
    m = re.search("|".join(DAYS), path.name.lower())
    if not m:
        raise ValueError(f"Cannot infer weekday from file name {path.name!r}")
    return m.group(0)


def _parse_ts(col: str) -> pl.Expr:
    return pl.coalesce(
        [pl.col(col).str.strptime(pl.Datetime("us"), f, strict=False) for f in _TS_FORMATS]
    )


def read_flow_csv(path: Path) -> pl.DataFrame:
    df = normalize_columns(pl.read_csv(path, infer_schema_length=100_000, encoding="utf8-lossy"))
    # The original release stores 'Infinity' / 'NaN' as text in some numeric columns.
    numeric_fix = [
        pl.col(c).cast(pl.Float64, strict=False)
        for c, t in df.schema.items()
        if t == pl.String and c not in _TEXT_COLUMNS
    ]
    df = df.with_columns(numeric_fix).with_columns(
        pl.col("label").str.strip_chars(),
        pl.lit(day_from_filename(path)).alias("day"),
    )
    if "timestamp" in df.columns:
        df = df.with_columns(_parse_ts("timestamp").alias("ts")).drop("timestamp")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("ts"))
    log.info("read %s: %d rows, %d columns", path.name, df.height, df.width)
    return df


def ingest(raw_dir: Path, out_path: Path) -> pl.DataFrame:
    files = sorted(raw_dir.glob("*.csv"), key=lambda p: DAYS.index(day_from_filename(p)))
    if not files:
        raise FileNotFoundError(f"No CSV files in {raw_dir}. See README 'Getting the data'.")
    df = pl.concat([read_flow_csv(f) for f in files], how="diagonal_relaxed")
    df = df.with_row_index("row_id").with_columns(pl.col("row_id").cast(pl.Int64))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    log.info("wrote %s: %d rows", out_path, df.height)
    return df
