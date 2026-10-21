"""Generalisation study end to end on three synthetic "networks" (no real datasets)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl

from sentinel.common.config import GeneralisationParams, Params
from sentinel.ids.generalisation import SHORTCUTS, Network, study


def _network(base: pl.DataFrame, name: str, shift: float, seed: int) -> Network:
    """The synthetic flows, with every numeric feature scaled (a different network)."""
    rng = np.random.default_rng(seed)
    num = [c for c, t in base.schema.items() if t.is_float() and c not in ("ts",)]
    df = base.with_columns(
        [(pl.col(c) * float(1 + shift * rng.random())).alias(c) for c in num[:20]]
    ).with_columns(
        pl.col("family").alias("category"), (pl.col("family") != "Benign").alias("is_attack")
    )
    return Network(name, df)


def test_study_trains_every_combination_and_reports(built: Params, sandbox: Any) -> None:
    base = pl.read_parquet(built.resolve(built.data.processed_dir) / "flows.parquet")
    nets = {
        "cic2017": _network(base, "cic2017", 0.0, 0),
        "unsw": _network(base, "unsw", 2.0, 1),
        "cic2018": _network(base, "cic2018", 0.5, 2),
    }
    params = built.model_copy(
        update={
            "generalisation": GeneralisationParams(
                train_benign=2000,
                train_attack=2000,
                val_benign=1000,
                val_attack=1000,
                test_benign=2000,
                test_attack=2000,
            )
        }
    )
    root = built.resolve(built.data.reports_dir).parent
    with sandbox(root):
        out = study(params, nets)
    assert len(out["runs"]) == 12  # 2 feature sets x (3 single + 3 pairs)
    assert not set(SHORTCUTS) & set(out["feature_sets"]["no_shortcuts"])
    for run in out["runs"]:
        for m in run["targets"].values():
            assert 0 <= m["roc_auc"] <= 1 and 0 <= m["attack_recall"] <= 1
    own = next(r for r in out["runs"] if r["sources"] == ["cic2017"] and r["features"] == "shared")
    assert own["targets"]["cic2017"]["roc_auc"] > 0.9  # in-domain sanity
    reports = built.resolve(built.data.reports_dir) / "ids"
    md = (reports / "generalisation.md").read_text(encoding="utf-8")
    assert "Without the shortcut features" in md and "**" in md
    headline = json.loads((reports / "generalisation_metrics.json").read_text(encoding="utf-8"))
    assert "no_shortcuts/unseen_unsw/two_networks" in headline
