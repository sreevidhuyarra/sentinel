"""Load the processed flows as model-ready arrays, one per split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from sentinel.common.config import Params
from sentinel.data.features import FeatureSpec
from sentinel.data.labels import FAMILIES

CLASSES: list[str] = FAMILIES
BENIGN = CLASSES.index("Benign")
Sample = Literal["full", "dev"]


@dataclass
class Split:
    X: np.ndarray  # float32, FeatureSpec-transformed
    y: np.ndarray  # int64 class index into CLASSES
    frame: pl.DataFrame  # untransformed rows (raw values for explanations, ids for alerts)


@dataclass
class Splits:
    train: Split
    val: Split
    test: Split
    spec: FeatureSpec

    @property
    def feature_names(self) -> list[str]:
        return self.spec.columns


def encode_labels(families: pl.Series) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(CLASSES)}
    return np.fromiter((lookup[f] for f in families), dtype=np.int64, count=len(families))


def load_splits(params: Params, sample: Sample = "full") -> Splits:
    processed = params.resolve(params.data.processed_dir)
    name = "flows.parquet" if sample == "full" else "flows_dev.parquet"
    df = pl.read_parquet(processed / name)
    spec = FeatureSpec.load(processed / "feature_spec.json")
    return splits_from_frame(df, spec)


def splits_from_frame(df: pl.DataFrame, spec: FeatureSpec) -> Splits:
    parts = {}
    for name in ("train", "val", "test"):
        part = df.filter(pl.col("split") == name)
        parts[name] = Split(spec.to_numpy(part), encode_labels(part["family"]), part)
    return Splits(parts["train"], parts["val"], parts["test"], spec)
