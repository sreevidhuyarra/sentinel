"""Data drift of live flows against the training reference (report section 9: Drift).

Reference: the benign part of the training sample. Against the full (22% attack) sample,
even attack-free traffic showed 22-33% of features drifted; against benign training
traffic, normal traffic shows 0-5% and attack-heavy windows 41-76% (measured on the test
split by day), so a threshold between them separates "normal" from "changed".

Two live windows:
- `all`: every sampled flow. Any change in the traffic mix, attack campaigns included.
- `benign`: flows the model passed as benign. Drift here is what the model cannot see:
  unknown attacks, or normal traffic that no longer looks like training normal.
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
from sqlalchemy import Engine, select

from sentinel.db.schema import flows

_COLUMN = re.compile(r"column=([^,)]+)")


def reference_frame(reference: pl.DataFrame, inputs: list[str], n: int, seed: int) -> pl.DataFrame:
    benign = reference.filter(pl.col("family") == "Benign").select(inputs)
    return benign.sample(min(n, len(benign)), seed=seed)


def live_window(
    engine: Engine, inputs: list[str], minutes: float, now: datetime | None = None
) -> dict[str, pl.DataFrame]:
    """Sampled flows scored in the last `minutes`: all of them, and those passed as benign."""
    now = now or datetime.now(UTC).replace(tzinfo=None)
    with engine.connect() as conn:
        rows = conn.execute(
            select(flows.c.features, flows.c.predicted_family).where(
                flows.c.scored_at >= now - timedelta(minutes=minutes)
            )
        ).all()
    feats = [r[0] if isinstance(r[0], dict) else json.loads(r[0]) for r in rows]
    frame = pl.DataFrame(
        [{c: f.get(c) for c in inputs} for f in feats], schema={c: pl.Float64 for c in inputs}
    )
    benign = pl.Series([r[1] == "Benign" for r in rows], dtype=pl.Boolean)
    return {"all": frame, "benign": frame.filter(benign) if len(frame) else frame}


def compute_drift(
    current: pl.DataFrame, reference: pl.DataFrame, method: str = "psi", threshold: float = 0.1
) -> dict[str, Any]:
    """Evidently data drift: share of drifted features and each feature's score."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    cols = [c for c in reference.columns if c in current.columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant columns: Evidently divides by zero
        snap = Report([DataDriftPreset(method=method, threshold=threshold)]).run(
            current_data=current.select(cols).to_pandas(),
            reference_data=reference.select(cols).to_pandas(),
        )
    metrics = snap.dict()["metrics"]
    overall = metrics[0]["value"]
    scores: dict[str, float] = {}
    for m in metrics[1:]:
        found = _COLUMN.search(str(m.get("metric_name", "")))
        if found and isinstance(m.get("value"), int | float):
            scores[found.group(1)] = float(m["value"])
    drifted = {c: v for c, v in scores.items() if v >= threshold}
    return {
        "n": len(current),
        "share": float(overall["share"]),
        "count": int(overall["count"]),
        "drifted": dict(sorted(drifted.items(), key=lambda kv: -kv[1])),
        "scores": scores,
    }
