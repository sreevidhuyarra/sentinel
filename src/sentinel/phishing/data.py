"""Stage: 11 public email corpora -> one deduplicated, leakage-safe split for fine-tuning.

Source: "Phishing Email Curated Datasets" (Champa, Rabbi & Zibran, 2024; Zenodo 8339691,
CC BY 4.0). Label 1 means *phishing* only for Nazario (phishing) and Nigerian (advance-fee
fraud); in the other corpora it means spam. The binary target is therefore "malicious or
unwanted vs legitimate", and a `kind` column (phishing / fraud / spam / legitimate) keeps
phishing recall reportable on its own.

Leakage control: exact duplicates (after normalisation) are removed, and emails are split
in groups (sender domain, or subject when there is no sender) so one sender's or one
campaign's mail never appears on both sides. The split is stratified by corpus x label.
"""

from __future__ import annotations

import json
import shutil
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

from sentinel.common.config import Params
from sentinel.common.logging import get_logger
from sentinel.phishing.text import dedup_key, extract_urls, model_input, sender_domain

log = get_logger(__name__)

ZENODO_RECORD = "https://zenodo.org/api/records/8339691"
MIN_TEXT_CHARS = 20
KAGGLE_COLUMNS = ["id", "text", "label", "kind", "source"]


def kind_of(source: str, label: int) -> str:
    if label == 0:
        return "legitimate"
    s = source.lower()
    if s.startswith("nazario"):
        return "phishing"
    if s.startswith("nigerian"):
        return "fraud"
    return "spam"


def _parse_date(value: Any) -> pd.Timestamp | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        ts = pd.Timestamp(parsedate_to_datetime(value))
        return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def read_corpus(path: Path) -> pd.DataFrame:
    """One corpus CSV with the columns every corpus shares (missing ones as None)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ("sender", "date", "subject", "body", "label"):
        if c not in df.columns:
            df[c] = None
    df = df[["sender", "date", "subject", "body", "label"]].copy()
    df["label"] = pd.to_numeric(df["label"].str.strip(), errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df["source"] = path.stem
    return df


def build(params: Params) -> dict[str, Any]:
    p = params.phishing
    raw = params.resolve(p.raw_dir)
    files = sorted(raw.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No corpus CSVs in {raw}; run `sentinel phishing download`.")
    frames = [read_corpus(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    rows_in = len(df)
    log.info("read %d emails from %d corpora", rows_in, len(files))

    df["text"] = [
        model_input(s, b, p.max_chars) for s, b in zip(df["subject"], df["body"], strict=True)
    ]
    df["urls"] = [
        extract_urls(f"{s or ''}\n{b or ''}")
        for s, b in zip(df["subject"], df["body"], strict=True)
    ]
    df["sender_domain"] = [sender_domain(s) for s in df["sender"]]
    df["date"] = [_parse_date(d) for d in df["date"]]
    df["kind"] = [kind_of(s, lab) for s, lab in zip(df["source"], df["label"], strict=True)]
    df = df[df["text"].str.len() >= MIN_TEXT_CHARS].copy()
    too_short = rows_in - len(df)

    # Exact duplicates: keep one; if the same text carries both labels, drop it entirely.
    df["dedup"] = df["text"].map(dedup_key)
    conflict = df.groupby("dedup")["label"].nunique()
    conflicted = set(conflict[conflict > 1].index)
    before = len(df)
    df = df[~df["dedup"].isin(conflicted)]
    df = df.sort_values(["sender_domain", "date"], na_position="last").drop_duplicates("dedup")
    dropped_dupes = before - len(df)

    subj_key = df["subject"].fillna("").map(lambda s: dedup_key(s)[:80])
    df["group"] = np.where(
        df["sender_domain"].notna(),
        "domain:" + df["sender_domain"].fillna(""),
        np.where(subj_key.str.len() > 0, "subject:" + subj_key, "row:" + df.index.astype(str)),
    )
    df = df.reset_index(drop=True)
    df["id"] = np.arange(len(df))

    strat = df["source"] + "_" + df["label"].astype(str)
    folds = np.empty(len(df), dtype=np.int64)
    sgkf = StratifiedGroupKFold(n_splits=p.n_folds, shuffle=True, random_state=params.seed)
    for k, (_, idx) in enumerate(sgkf.split(df, strat, groups=df["group"])):
        folds[idx] = k
    df["split"] = np.select(
        [folds < p.n_folds - 2, folds == p.n_folds - 2], ["train", "val"], default="test"
    )

    out = pl.from_pandas(
        df[
            [
                "id",
                "source",
                "kind",
                "label",
                "text",
                "urls",
                "sender_domain",
                "date",
                "group",
                "split",
            ]
        ]
    )
    processed = params.resolve(p.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    out.write_parquet(processed / "emails.parquet", compression="zstd")

    kaggle = params.resolve(p.kaggle_dir)
    if kaggle.exists():
        shutil.rmtree(kaggle)
    kaggle.mkdir(parents=True)
    for split in ("train", "val", "test"):
        out.filter(pl.col("split") == split).select(KAGGLE_COLUMNS).write_parquet(
            kaggle / f"{split}.parquet", compression="zstd"
        )
    _write_kaggle_metadata(kaggle)

    summary = _summary(out, rows_in, too_short, dropped_dupes, len(conflicted))
    reports = params.resolve(params.data.reports_dir) / "phishing"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "data_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def _summary(
    out: pl.DataFrame, rows_in: int, too_short: int, dupes: int, conflicts: int
) -> dict[str, Any]:
    by = out.group_by(["split", "kind"]).len().sort(["split", "kind"])
    table: dict[str, dict[str, int]] = {}
    for split, kind, n in by.iter_rows():
        table.setdefault(kind, {})[split] = n
    groups_crossing = (
        out.group_by("group")
        .agg(pl.col("split").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    return {
        "rows_in": rows_in,
        "dropped_too_short": too_short,
        "dropped_duplicates": dupes,
        "conflicting_label_texts": conflicts,
        "rows_out": out.height,
        "groups": out["group"].n_unique(),
        "groups_in_more_than_one_split": groups_crossing,
        "by_kind": table,
        "by_source": dict(out.group_by("source").len().sort("source").iter_rows()),
    }


def _write_kaggle_metadata(folder: Path) -> None:
    meta = {
        "title": "Sentinel phishing emails",
        "id": "YOUR_KAGGLE_USERNAME/sentinel-phishing-emails",
        "licenses": [{"name": "CC-BY-4.0"}],
    }
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (folder / "README.md").write_text(
        "# Sentinel phishing emails\n\n"
        "Deduplicated, leakage-safe train/val/test split of the Phishing Email Curated Datasets "
        "(Champa, Rabbi & Zibran, 2024; https://doi.org/10.5281/zenodo.8339691, CC BY 4.0).\n\n"
        "Columns: id, text (subject + body, HTML removed, URLs replaced by [URL]), "
        "label (1 = phishing / fraud / spam, 0 = legitimate), kind, source.\n",
        encoding="utf-8",
    )


def download(raw_dir: Path) -> list[Path]:
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_RECORD) as resp:
        record = json.load(resp)
    paths = []
    for f in record["files"]:
        target = raw_dir / f["key"]
        if not target.exists():
            log.info("downloading %s", f["key"])
            with urllib.request.urlopen(f["links"]["self"]) as resp, target.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
        paths.append(target)
    return paths
