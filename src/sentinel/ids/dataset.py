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


def with_extra_train(s: Splits, extra: pl.DataFrame) -> Splits:
    """Append labelled rows (e.g. live flows labelled by analysts) to the training split only.

    `extra` needs the spec's input columns and a `family` column. Validation and test stay
    untouched, so a retrained model is compared with production on the same held-out data.
    """
    if extra.is_empty():
        return s
    cols = [c for c in s.train.frame.columns if c in extra.columns]
    frame = pl.concat([s.train.frame.select(cols), extra.select(cols)], how="vertical_relaxed")
    train = Split(
        np.concatenate([s.train.X, s.spec.to_numpy(extra)]),
        np.concatenate([s.train.y, encode_labels(extra["family"])]),
        frame,
    )
    return Splits(train, s.val, s.test, s.spec)


def _row_keys(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X)
    return X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1]))).ravel()


def without_overlap(s: Splits, extra: pl.DataFrame) -> tuple[Splits, dict[str, int]]:
    """Drop validation and test rows identical (in model inputs) to a labelled live flow.

    Live traffic can repeat held-out flows (the replay demo streams the test split). A
    retrained model must never be tuned or scored on flows it was trained on, so those rows
    leave validation and test before retraining, and production is re-scored on the same
    reduced test set (`train`). Returns the splits and how many rows each lost.
    """
    if extra.is_empty():
        return s, {"val": 0, "test": 0}
    live = _row_keys(s.spec.to_numpy(extra))
    parts, dropped = {}, {}
    for name in ("val", "test"):
        part: Split = getattr(s, name)
        keep = ~np.isin(_row_keys(part.X), live)
        dropped[name] = int((~keep).sum())
        parts[name] = Split(part.X[keep], part.y[keep], part.frame.filter(pl.Series(keep)))
    return Splits(s.train, parts["val"], parts["test"], s.spec), dropped


def splits_from_frame(df: pl.DataFrame, spec: FeatureSpec) -> Splits:
    parts = {}
    for name in ("train", "val", "test"):
        part = df.filter(pl.col("split") == name)
        parts[name] = Split(spec.to_numpy(part), encode_labels(part["family"]), part)
    return Splits(parts["train"], parts["val"], parts["test"], spec)
