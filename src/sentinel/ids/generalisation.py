"""Generalisation study: does the detector work on networks it was not trained on?

Two experiments, designed before any target result was seen:

1. **Shortcut features.** Retrain without four features that describe how a testbed was
   configured rather than how attacks behave: `dst_port` (which services ran where) and
   `fwd_init_win_bytes`, `bwd_init_win_bytes`, `fwd_seg_size_min` (TCP stack defaults of
   the operating systems). Module 1 found `dst_port` and `bwd_init_win_bytes` dominating
   whole families on CIC-IDS2017.
2. **More than one network.** Train on one network or on two together; test on all three.

Networks (all extracted with CICFlowMeter, so they share most features):
- CIC-IDS2017 (DistriNet corrected release): the project's training data.
- CIC-UNSW-NB15: UNSW's testbed, a different lab with different attack tools.
- CSE-CIC-IDS2018 (DistriNet corrected release): a different, larger network built by the
  same lab with mostly the same attack tools, so an easier test than UNSW.

The task is benign vs attack (the three use different attack categories). Each network is
split in time (train / val / test per label); models fit on the training networks' train
split, stop early and pick their 1%-false-alarm threshold on those networks' validation
split, and are scored on every network's test split. Nothing is chosen on a target.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from sentinel.common.config import Params, get_settings
from sentinel.common.logging import get_logger
from sentinel.data.clean import clean
from sentinel.data.columns import ID_COLUMNS, LABEL_COLUMNS, canonical, normalize_columns
from sentinel.data.features import FeatureSpec, add_derived
from sentinel.data.ingest import _parse_ts
from sentinel.data.labels import is_attempted, normalize_label
from sentinel.data.splits import temporal_split
from sentinel.ids.crossdataset import load_unsw, shared_spec

log = get_logger(__name__)

SHORTCUTS = ("dst_port", "fwd_init_win_bytes", "bwd_init_win_bytes", "fwd_seg_size_min")
NETWORKS = ("cic2017", "unsw", "cic2018")
NAMES = {"cic2017": "CIC-IDS2017", "unsw": "UNSW-NB15", "cic2018": "CSE-CIC-IDS2018"}
TARGET_FPR = 0.01
_TEXT = frozenset(ID_COLUMNS + LABEL_COLUMNS)


@dataclass
class Network:
    """One network's flows: features, `is_attack`, `category` (its own label) and `split`."""

    name: str
    frame: pl.DataFrame


# --- CSE-CIC-IDS2018 -----------------------------------------------------------------------


def _is_attack(raw: str) -> bool:
    """Attack unless benign; "Attempted" flows (no payload delivered) count as benign, as in
    the CIC-IDS2017 pipeline (`data.attempted_policy: benign`)."""
    return normalize_label(raw) != "benign" and not is_attempted(raw)


def _sample_csv(path: Path, per_label: int, seed: int) -> pl.DataFrame:
    """Up to `per_label` flows of each label from one CSV, without loading the whole file."""
    names = pl.scan_csv(path, infer_schema=False).collect_schema().names()
    label = next(c for c in names if canonical(c) == "label")
    counts = dict(
        pl.scan_csv(path, infer_schema=False)
        .group_by(label)
        .len()
        .collect(engine="streaming")
        .iter_rows()
    )
    keep = pl.DataFrame(
        {label: list(counts), "_p": [min(1.0, per_label / n) for n in counts.values()]}
    )
    sampled = (
        pl.scan_csv(path, infer_schema=False)
        .with_row_index("_i")
        .join(keep.lazy(), on=label)
        .filter((pl.col("_i").hash(seed) % 1_000_000) < pl.col("_p") * 1_000_000)
        .drop("_i", "_p")
        .collect(engine="streaming")
    )
    df = normalize_columns(sampled)
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in df.columns if c not in _TEXT]
    ).with_columns(pl.col("label").str.strip_chars())
    if "timestamp" in df.columns:
        df = df.with_columns(_parse_ts("timestamp").alias("ts")).drop("timestamp")
    log.info(
        "%s: %s labels, kept %d of %d flows",
        path.name,
        len(counts),
        df.height,
        sum(counts.values()),
    )
    return df


def load_cic2018(zip_path: Path, cache: Path, per_label: int, seed: int) -> pl.DataFrame:
    """CSE-CIC-IDS2018 (corrected), sampled per label per day file, cleaned like CIC-IDS2017.

    The CSVs are tens of GB unpacked, so each is extracted, sampled and deleted in turn.
    """
    if cache.exists():
        return pl.read_parquet(cache)
    tmp = cache.parent / "_cic2018_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in sorted(m for m in zf.namelist() if m.lower().endswith(".csv")):
            out = tmp / Path(member).name
            with zf.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=16 << 20)
            try:
                # `day` is an identifier column, so cleaning never treats it as a feature.
                parts.append(
                    _sample_csv(out, per_label, seed).with_columns(
                        pl.lit(Path(member).stem).alias("day")
                    )
                )
            finally:
                out.unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    df = pl.concat(parts, how="diagonal_relaxed").with_row_index("row_id")
    df = df.with_columns(pl.col("row_id").cast(pl.Int64))
    df, stats = clean(df)
    log.info("CSE-CIC-IDS2018 sample: %s", stats.as_dict())
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache, compression="zstd")
    return df


# --- the three networks ----------------------------------------------------------------------


def _labelled(df: pl.DataFrame, category: str, attack: pl.Expr) -> pl.DataFrame:
    return df.with_columns(pl.col(category).alias("category"), attack.alias("is_attack"))


def load_networks(params: Params, unsw_csv: Path, cic2018_zip: Path) -> dict[str, Network]:
    g = params.generalisation
    processed = params.resolve(params.data.processed_dir)
    interim = params.resolve(params.data.interim_dir)
    cic17 = pl.read_parquet(processed / "flows.parquet")
    cic17 = _labelled(cic17, "family", pl.col("family") != "Benign")

    unsw = load_unsw(unsw_csv, interim / "cic_unsw_nb15.parquet")
    unsw = temporal_split(unsw, 0.6, 0.2, gap_rows=50, group_cols=("label",))
    unsw = _labelled(unsw, "label", pl.col("label") != "Benign")

    cic18 = load_cic2018(
        cic2018_zip, interim / "cse_cic_ids2018.parquet", g.cic2018_per_label, params.seed
    )
    labels = cic18["label"].unique().to_list()
    cic18 = cic18.with_columns(
        pl.col("label").replace_strict({x: _is_attack(x) for x in labels}).alias("is_attack"),
        pl.col("label").replace_strict({x: normalize_label(x) for x in labels}).alias("category"),
    )
    cic18 = temporal_split(cic18, 0.6, 0.2, gap_rows=50, group_cols=("label",))
    return {
        "cic2017": Network("cic2017", cic17),
        "unsw": Network("unsw", unsw),
        "cic2018": Network("cic2018", cic18),
    }


def feature_sets(spec: FeatureSpec, networks: dict[str, Network]) -> dict[str, FeatureSpec]:
    """Features every network has; and the same without the shortcut features."""
    available = set.intersection(
        *(set(add_derived(n.frame.head(0)).columns) for n in networks.values())
    )
    shared = shared_spec(spec, available)
    drop = set(SHORTCUTS)
    no_shortcuts = FeatureSpec(
        columns=[c for c in shared.columns if c not in drop],
        log_columns=[c for c in shared.log_columns if c not in drop],
        dropped_constant=shared.dropped_constant,
    )
    return {"shared": shared, "no_shortcuts": no_shortcuts}


# --- training and scoring ---------------------------------------------------------------------


def _sample(df: pl.DataFrame, n_benign: int, n_attack: int, seed: int) -> pl.DataFrame:
    """Benign rows up to `n_benign`; attack rows up to `n_attack`, spread over categories
    (each category gets an equal cap, so a flood of one attack does not crowd out the rest)."""
    benign = df.filter(~pl.col("is_attack"))
    benign = benign.sample(min(n_benign, benign.height), seed=seed)
    attacks = df.filter(pl.col("is_attack"))
    cats = attacks["category"].n_unique() or 1
    cap = max(1, n_attack // cats)
    attacks = attacks.filter(pl.int_range(pl.len()).shuffle(seed).over("category") < cap)
    return pl.concat([benign, attacks], how="diagonal_relaxed")


def _arrays(
    nets: list[Network], split: str, spec: FeatureSpec, n_benign: int, n_attack: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, labels and weights: every network weighs the same, and within a network
    benign and attack weigh the same, so neither the biggest network nor class balance
    decides what the model learns."""
    X, y, w = [], [], []
    for net in nets:
        part = _sample(net.frame.filter(pl.col("split") == split), n_benign, n_attack, seed)
        lab = part["is_attack"].to_numpy().astype(np.int64)
        n_pos, n_neg = max(int(lab.sum()), 1), max(int((1 - lab).sum()), 1)
        X.append(spec.to_numpy(part))
        y.append(lab)
        w.append(np.where(lab == 1, 0.5 / n_pos, 0.5 / n_neg))
    weights = np.concatenate(w)
    # Each network's weights sum to 1; rescale so the average row weighs 1, which keeps
    # LightGBM's min_child_samples / hessian limits meaning what they say.
    return np.concatenate(X), np.concatenate(y), weights * len(weights) / len(nets)


def fit(sources: list[Network], spec: FeatureSpec, params: Params) -> tuple[lgb.Booster, float]:
    g = params.generalisation
    X, y, w = _arrays(sources, "train", spec, g.train_benign, g.train_attack, params.seed)
    Xv, yv, wv = _arrays(sources, "val", spec, g.val_benign, g.val_attack, params.seed)
    booster = lgb.train(
        {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": params.seed,
            "verbosity": -1,
            "metric": "auc",
        },
        lgb.Dataset(X, y, weight=w),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(Xv, yv, weight=wv)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    val = np.asarray(booster.predict(Xv, num_iteration=booster.best_iteration))
    # 1% of the training networks' benign validation flows exceed the threshold.
    threshold = float(np.nextafter(np.quantile(val[yv == 0], 1 - TARGET_FPR), np.inf))
    return booster, threshold


def score(
    booster: lgb.Booster, threshold: float, target: Network, spec: FeatureSpec, params: Params
) -> dict[str, Any]:
    g = params.generalisation
    test = _sample(
        target.frame.filter(pl.col("split") == "test"),
        g.test_benign,
        g.test_attack,
        params.seed,
    )
    s = np.asarray(booster.predict(spec.to_numpy(test), num_iteration=booster.best_iteration))
    is_attack = test["is_attack"].to_numpy()
    flagged = s >= threshold
    own = float(np.nextafter(np.quantile(s[~is_attack], 1 - TARGET_FPR), np.inf))
    cats = test["category"].to_numpy()
    return {
        "flows": len(s),
        "roc_auc": float(roc_auc_score(is_attack, s)),
        "pr_auc": float(average_precision_score(is_attack, s)),
        "benign_fpr": float(flagged[~is_attack].mean()),
        "attack_recall": float(flagged[is_attack].mean()),
        # Re-thresholded on the target's own benign traffic: what calibration alone buys.
        "recall_at_1pct_fpr_target_threshold": float((s[is_attack] >= own).mean()),
        "per_category_recall": {
            str(c): float(flagged[cats == c].mean()) for c in sorted(set(cats[is_attack].tolist()))
        },
    }


def run(params: Params, unsw_csv: Path, cic2018_zip: Path) -> dict[str, Any]:
    return study(params, load_networks(params, unsw_csv, cic2018_zip))


def study(params: Params, networks: dict[str, Network]) -> dict[str, Any]:
    """Both experiments on already-loaded networks (keys: NETWORKS); writes the reports."""
    spec = FeatureSpec.load(params.resolve(params.data.processed_dir) / "feature_spec.json")
    sets = feature_sets(spec, networks)
    source_sets = [(n,) for n in NETWORKS] + [
        tuple(n for n in NETWORKS if n != held) for held in NETWORKS
    ]
    results: dict[str, Any] = {
        "networks": {
            k: {
                "flows": n.frame.height,
                "attack_share": float(n.frame["is_attack"].to_numpy().mean()),
                "test_flows": n.frame.filter(pl.col("split") == "test").height,
            }
            for k, n in networks.items()
        },
        "feature_sets": {k: v.columns for k, v in sets.items()},
        "shortcuts": list(SHORTCUTS),
        "runs": [],
    }
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("sentinel-ids-generalisation")
    with mlflow.start_run(run_name="generalisation"):
        for fs_name, fs in sets.items():
            for sources in source_sets:
                booster, t = fit([networks[s] for s in sources], fs, params)
                gain = booster.feature_importance("gain").astype(np.float64)
                top = np.argsort(-gain)[:8]
                targets = {
                    name: score(booster, t, net, fs, params) for name, net in networks.items()
                }
                entry: dict[str, Any] = {
                    "features": fs_name,
                    "sources": list(sources),
                    "threshold": t,
                    "trees": booster.best_iteration,
                    "top_features": [
                        {"feature": fs.columns[i], "gain_share": float(gain[i] / gain.sum())}
                        for i in top
                    ],
                    "targets": targets,
                }
                results["runs"].append(entry)
                tag = f"{fs_name}_{'-'.join(sources)}"  # MLflow names allow no '+'
                for name, m in targets.items():
                    mlflow.log_metric(f"{tag}_to_{name}_roc_auc", m["roc_auc"])
                    mlflow.log_metric(f"{tag}_to_{name}_recall", m["attack_recall"])
                log.info(
                    "%s from %s: %s",
                    fs_name,
                    "+".join(sources),
                    {k: round(v["roc_auc"], 3) for k, v in targets.items()},
                )
        out = params.resolve(params.data.reports_dir) / "ids"
        out.mkdir(parents=True, exist_ok=True)
        (out / "generalisation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out / "generalisation.md").write_text(report(results), encoding="utf-8")
        (out / "generalisation_metrics.json").write_text(
            json.dumps(_headline(results), indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(out / "generalisation.md"))
    return results


# --- report ---------------------------------------------------------------------------------


def _find(r: dict[str, Any], fs: str, sources: tuple[str, ...]) -> dict[str, Any]:
    return next(x for x in r["runs"] if x["features"] == fs and tuple(x["sources"]) == sources)


def _headline(r: dict[str, Any]) -> dict[str, Any]:
    """ROC-AUC on each network when it was not trained on (DVC metrics)."""
    out: dict[str, Any] = {}
    for fs in ("shared", "no_shortcuts"):
        for held in NETWORKS:
            others = tuple(n for n in NETWORKS if n != held)
            out[f"{fs}/unseen_{held}/two_networks"] = _find(r, fs, others)["targets"][held][
                "roc_auc"
            ]
            out[f"{fs}/unseen_{held}/best_single"] = max(
                _find(r, fs, (s,))["targets"][held]["roc_auc"] for s in others
            )
    return out


def report(r: dict[str, Any]) -> str:
    def src(sources: list[str]) -> str:
        return " + ".join(NAMES[s] for s in sources)

    lines = [
        "# Generalisation: training on one or two networks, testing on another\n",
        "Benign vs attack, LightGBM. Thresholds come from the training networks' validation "
        "split (1% false alarms there); nothing is chosen on a target. Cells on the "
        "diagonal (a network's own test split) are in-domain; the rest are unseen networks.\n",
        "| Network | Flows used | Attack share | Test flows |",
        "|---|---|---|---|",
    ]
    for k, n in r["networks"].items():
        lines.append(
            f"| {NAMES[k]} | {n['flows']:,} | {n['attack_share']:.1%} | {n['test_flows']:,} |"
        )
    for fs, title in (
        ("shared", "All shared features"),
        ("no_shortcuts", "Without the shortcut features"),
    ):
        n_feat = len(r["feature_sets"][fs])
        lines += [
            f"\n## {title} ({n_feat} features)\n",
            "ROC-AUC (attack recall at the training threshold, false alarms) per test network.\n",
            "| Trained on | " + " | ".join(NAMES[t] for t in NETWORKS) + " |",
            "|---|" + "---|" * len(NETWORKS),
        ]
        for run in (x for x in r["runs"] if x["features"] == fs):
            cells = []
            for t in NETWORKS:
                m = run["targets"][t]
                seen = t in run["sources"]
                cell = (
                    f"{m['roc_auc']:.3f} ({m['attack_recall']:.0%} caught, "
                    f"{m['benign_fpr']:.1%} FA)"
                )
                cells.append(cell if seen else f"**{cell}**")
            lines.append(f"| {src(run['sources'])} | " + " | ".join(cells) + " |")
    lines += [
        "\nBold: a network the model never saw. FA: share of that network's benign test "
        "flows flagged.\n",
        "## Effect of removing the shortcut features (ROC-AUC on the unseen network)\n",
        "| Trained on | Unseen network | All shared | Without shortcuts | Change |",
        "|---|---|---|---|---|",
    ]
    for run in (x for x in r["runs"] if x["features"] == "shared"):
        other = _find(r, "no_shortcuts", tuple(run["sources"]))
        for t in NETWORKS:
            if t in run["sources"]:
                continue
            a, b = run["targets"][t]["roc_auc"], other["targets"][t]["roc_auc"]
            lines.append(
                f"| {src(run['sources'])} | {NAMES[t]} | {a:.3f} | {b:.3f} | {b - a:+.3f} |"
            )
    lines += [
        "\n## What the models rely on (top features by gain share)\n",
        "| Features | Trained on | Top features |",
        "|---|---|---|",
    ]
    for run in r["runs"]:
        top = ", ".join(f"`{f['feature']}` {f['gain_share']:.0%}" for f in run["top_features"][:5])
        lines.append(f"| {run['features']} | {src(run['sources'])} | {top} |")
    return "\n".join(lines) + "\n"
