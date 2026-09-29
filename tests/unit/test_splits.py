from datetime import datetime, timedelta

import polars as pl
import pytest

from sentinel.data.splits import holdout_family_split, random_split, temporal_split


def _flows(n_benign: int = 1000, n_attack: int = 500, n_rare: int = 11) -> pl.DataFrame:
    rows = []
    t0 = datetime(2017, 7, 5, 9)
    for label, family, n in [
        ("BENIGN", "Benign", n_benign),
        ("DoS Hulk", "DoS", n_attack),
        ("Heartbleed", "DoS", n_rare),
    ]:
        rows += [
            {"label": label, "family": family, "day": "wednesday", "ts": t0 + timedelta(seconds=i)}
            for i in range(n)
        ]
    return pl.DataFrame(rows).with_row_index("row_id")


def test_temporal_split_is_chronological_within_each_label() -> None:
    out = temporal_split(_flows(), gap_rows=10)
    for _, g in out.group_by("label"):
        tr, va, te = (g.filter(pl.col("split") == s)["ts"] for s in ("train", "val", "test"))
        assert tr.max() < va.min() and va.max() < te.min()


def test_gap_rows_removed_at_each_boundary() -> None:
    out = temporal_split(_flows(n_rare=0), gap_rows=10)
    # 1000 benign -> gap 10 (cap), 500 attack -> gap 10 (2% of 500)
    assert out.height == 1500 - 2 * 10 - 2 * 10


def test_rare_class_present_in_every_split() -> None:
    out = temporal_split(_flows(), gap_rows=50)
    rare = out.filter(pl.col("label") == "Heartbleed")
    assert set(rare["split"]) == {"train", "val", "test"}
    assert rare.height == 11


def test_bad_fractions_rejected() -> None:
    with pytest.raises(ValueError):
        temporal_split(_flows(), train_frac=0.8, val_frac=0.3)


def test_holdout_family_only_in_test() -> None:
    df = _flows()
    out = holdout_family_split(df, "DoS")
    assert set(out.filter(pl.col("family") == "DoS")["split"]) == {"test"}
    assert out.filter(pl.col("family") == "DoS").height == 511


def test_random_split_covers_all_rows() -> None:
    out = random_split(_flows())
    assert out.height == 1511
    assert set(out["split"]) == {"train", "val", "test"}
