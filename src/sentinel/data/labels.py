"""Map fine-grained CIC-IDS2017 labels (original and corrected releases) to attack families."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

import polars as pl


class Family(StrEnum):
    BENIGN = "Benign"
    DOS = "DoS"
    DDOS = "DDoS"
    PORTSCAN = "PortScan"
    BRUTEFORCE = "BruteForce"
    WEBATTACK = "WebAttack"
    BOT = "Bot"
    INFILTRATION = "Infiltration"


FAMILIES: list[str] = [f.value for f in Family]

AttemptedPolicy = Literal["benign", "drop", "keep"]

_ATTEMPTED = re.compile(r"\battempted\b")


def normalize_label(raw: str) -> str:
    """Lowercase and collapse punctuation. The original CSVs spell the web-attack
    dash as a mis-decoded byte (e.g. 'Web Attack � Brute Force')."""
    return re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()


def is_attempted(raw: str) -> bool:
    return bool(_ATTEMPTED.search(normalize_label(raw)))


def to_family(raw: str) -> Family:
    """Family of a fine label, ignoring any '- Attempted' suffix. Order matters:
    'ddos' before 'dos', 'web attack brute force' before 'brute force', and portscan
    before infiltration: the corrected release's 'Infiltration - Portscan' (~72k flows,
    a scan launched from the compromised host) behaves like a port scan, while true
    infiltration traffic is only a few dozen flows."""
    s = _ATTEMPTED.sub("", normalize_label(raw)).strip()
    if s == "benign":
        return Family.BENIGN
    if s.startswith("ddos"):
        return Family.DDOS
    if s.startswith("dos") or s.startswith("heartbleed"):
        return Family.DOS
    if s.startswith("web attack"):
        return Family.WEBATTACK
    if "portscan" in s or "port scan" in s:
        return Family.PORTSCAN
    if s.startswith("infiltration"):
        return Family.INFILTRATION
    if "patator" in s or "brute force" in s:
        return Family.BRUTEFORCE
    if s.startswith("bot"):
        return Family.BOT
    raise ValueError(f"Unmapped label: {raw!r}")


def add_family_columns(df: pl.DataFrame, policy: AttemptedPolicy = "benign") -> pl.DataFrame:
    """Add `family` and `attempted` columns from the raw `label` column.

    policy="benign" relabels attempted flows as Benign (they carry no attack payload),
    "drop" removes them, "keep" leaves them in their attack family.
    """
    mapping = pl.DataFrame(
        {"label": df["label"].unique().to_list()},
        schema={"label": pl.String},
    ).with_columns(
        pl.col("label")
        .map_elements(lambda s: to_family(s).value, return_dtype=pl.String)
        .alias("family"),
        pl.col("label").map_elements(is_attempted, return_dtype=pl.Boolean).alias("attempted"),
    )
    out = df.join(mapping, on="label", how="left")
    if policy == "drop":
        return out.filter(~pl.col("attempted"))
    if policy == "benign":
        return out.with_columns(
            pl.when(pl.col("attempted"))
            .then(pl.lit(Family.BENIGN.value))
            .otherwise(pl.col("family"))
            .alias("family")
        )
    return out
