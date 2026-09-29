"""Feature pipeline fitted on the training split and saved as JSON.

The same FeatureSpec is loaded by training jobs and by the live detector, so a flow is
transformed identically offline and online. Scaling for neural models lives with those
models; this spec only selects columns, derives ratios and log-compresses heavy tails.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from sentinel.data.columns import feature_columns

# Categorical-like codes where a log would destroy meaning.
_NO_LOG = frozenset({"dst_port", "protocol", "icmp_code", "icmp_type"})
_HEAVY_TAIL_MAX = 1_000.0
DERIVED = ("fwd_bwd_bytes_ratio", "fwd_bwd_packets_ratio")


def add_derived(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("total_length_of_fwd_packet") / (pl.col("total_length_of_bwd_packet") + 1)).alias(
            "fwd_bwd_bytes_ratio"
        ),
        (pl.col("total_fwd_packet") / (pl.col("total_bwd_packets") + 1)).alias(
            "fwd_bwd_packets_ratio"
        ),
    )


def _signed_log1p(col: str) -> pl.Expr:
    return (pl.col(col).sign() * pl.col(col).abs().log1p()).alias(col)


@dataclass
class FeatureSpec:
    columns: list[str]
    log_columns: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls, train: pl.DataFrame, log_transform: bool = True, drop_constant: bool = True
    ) -> FeatureSpec:
        train = add_derived(train)
        cols = feature_columns(train)
        constant = [c for c in cols if train[c].n_unique() <= 1] if drop_constant else []
        cols = [c for c in cols if c not in constant]
        log_cols: list[str] = []
        if log_transform:
            maxabs = train.select([pl.col(c).abs().max().alias(c) for c in cols]).row(0, named=True)
            log_cols = [c for c in cols if c not in _NO_LOG and float(maxabs[c]) > _HEAVY_TAIL_MAX]
        return cls(columns=cols, log_columns=log_cols, dropped_constant=constant)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        missing = set(self.columns) - set(add_derived(df.head(0)).columns)
        if missing:
            raise KeyError(f"Input is missing feature columns: {sorted(missing)}")
        out = (
            add_derived(df)
            .select([pl.col(c).cast(pl.Float64) for c in self.columns])
            .with_columns([_signed_log1p(c) for c in self.log_columns])
        )
        return out.with_columns(pl.all().fill_nan(0).fill_null(0).cast(pl.Float32))

    @property
    def input_columns(self) -> list[str]:
        """Columns a caller must supply (derived features are computed here)."""
        base = {
            "total_length_of_fwd_packet",
            "total_length_of_bwd_packet",
            "total_fwd_packet",
            "total_bwd_packets",
        }
        return sorted((set(self.columns) - set(DERIVED)) | base)

    def raw(self, df: pl.DataFrame) -> np.ndarray:
        """Feature values before log compression, for human-readable explanations."""
        return (
            add_derived(df)
            .select([pl.col(c).cast(pl.Float64) for c in self.columns])
            .fill_nan(0)
            .fill_null(0)
            .to_numpy()
        )

    def to_numpy(self, df: pl.DataFrame) -> np.ndarray:
        return self.transform(df).to_numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeatureSpec:
        return cls(**json.loads(path.read_text(encoding="utf-8")))
