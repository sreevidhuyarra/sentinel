"""Cross-dataset test: train on CIC-IDS2017, score UNSW-NB15 without any retuning (goal G3).

Target data is CIC-UNSW-NB15 (the UNSW-NB15 packet captures re-extracted with
CICFlowMeter by the Canadian Institute for Cybersecurity), so both datasets share one
feature extractor and most of our features. Columns the target lacks (the corrected
CIC-IDS2017 release adds ICMP code / type, total TCP flow time and RST flag counts) are
dropped; models are retrained without them and their in-domain test score is reported
so the cost of dropping them is visible.

UNSW-NB15 uses different attack categories, so the headline is benign-vs-attack:
detection rate per UNSW category, benign false-positive rate, and threshold-free ranking
quality (ROC-AUC, recall at 1% FPR with a threshold chosen on UNSW itself). A model
trained on UNSW is reported as the in-domain reference: the gap between the two is the
cost of moving to a new network. A per-feature shift table explains where it comes from.
"""

from __future__ import annotations

import json
import time
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
from sentinel.data.columns import ID_COLUMNS, LABEL_COLUMNS, normalize_columns
from sentinel.data.features import FeatureSpec, add_derived
from sentinel.data.splits import temporal_split
from sentinel.ids.dataset import BENIGN, CLASSES, splits_from_frame
from sentinel.ids.decision import attack_score, decide
from sentinel.ids.metrics import evaluate
from sentinel.ids.models import LogitModel
from sentinel.ids.models.gbm import LightGBMModel, fit_lightgbm, fit_xgboost
from sentinel.ids.models.mlp import fit_mlp
from sentinel.ids.train import Scored, choose_ensemble_weight, score_model
from sentinel.ids.weights import class_weights, sample_weights

log = get_logger(__name__)

TARGET_FPR = 0.01
# UNSW-NB15 categories with a counterpart among our families; the rest have none.
FAMILY_MATCH = {"DoS": "DoS", "Reconnaissance": "PortScan"}
_TEXT = frozenset(ID_COLUMNS + LABEL_COLUMNS)
_UNSW_TS = "%d/%m/%Y %I:%M:%S %p"


def load_unsw(csv_path: Path, cache: Path | None = None) -> pl.DataFrame:
    """CIC-UNSW-NB15 flows with canonical columns, cleaned exactly like CIC-IDS2017."""
    if cache is not None and cache.exists():
        return pl.read_parquet(cache)
    df = normalize_columns(
        pl.read_csv(csv_path, infer_schema_length=100_000, encoding="utf8-lossy")
    )
    df = df.with_columns(
        [
            pl.col(c).cast(pl.Float64, strict=False)
            for c, t in df.schema.items()
            if t == pl.String and c not in _TEXT
        ]
    ).with_columns(
        pl.col("label").str.strip_chars(),
        pl.col("timestamp").str.strptime(pl.Datetime("us"), _UNSW_TS, strict=False).alias("ts"),
    )
    df = df.drop("timestamp").with_row_index("row_id")
    df = df.with_columns(pl.col("row_id").cast(pl.Int64))
    df, stats = clean(df)
    log.info("CIC-UNSW-NB15: %s", stats.as_dict())
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache, compression="zstd")
    return df


def shared_spec(spec: FeatureSpec, available: set[str]) -> FeatureSpec:
    """The spec restricted to features computable from the target's columns."""
    drop = set(spec.columns) - available
    return FeatureSpec(
        columns=[c for c in spec.columns if c not in drop],
        log_columns=[c for c in spec.log_columns if c not in drop],
        dropped_constant=spec.dropped_constant,
    )


def recall_at_fpr(is_attack: np.ndarray, score: np.ndarray, fpr: float) -> tuple[float, float]:
    """Attack recall when the threshold is set so that `fpr` of benign flows exceed it.
    Uses the target's own labels: an upper bound on what re-thresholding alone could buy."""
    t = float(np.quantile(score[~is_attack], 1 - fpr))
    t = float(np.nextafter(t, np.inf))
    return float((score[is_attack] >= t).mean()), t


def transfer_metrics(proba: np.ndarray, threshold: float, labels: pl.Series) -> dict[str, Any]:
    """Binary (benign vs any attack) metrics of a CIC-trained model on the target data."""
    lab = labels.to_numpy()
    is_attack = lab != "Benign"
    score = attack_score(proba)
    pred = decide(proba, threshold)
    flagged = pred != BENIGN
    tp = int((flagged & is_attack).sum())
    fp = int((flagged & ~is_attack).sum())
    precision = tp / max(tp + fp, 1)
    recall = float(flagged[is_attack].mean())
    oracle_recall, oracle_t = recall_at_fpr(is_attack, score, TARGET_FPR)
    per_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted(set(lab)):
        m = lab == cat
        entry: dict[str, Any] = {"flows": int(m.sum()), "flagged": float(flagged[m].mean())}
        if cat in FAMILY_MATCH:
            want = CLASSES.index(FAMILY_MATCH[cat])
            det = m & flagged
            entry["as_" + FAMILY_MATCH[cat]] = (
                float((pred[det] == want).mean()) if det.any() else 0.0
            )
        if cat != "Benign":
            counts = np.bincount(pred[m], minlength=len(CLASSES))
            entry["predicted_as"] = {CLASSES[i]: int(n) for i, n in enumerate(counts) if n}
        per_cat[cat] = entry
    return {
        "benign_fpr": float(flagged[~is_attack].mean()),
        "attack_recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "roc_auc": float(roc_auc_score(is_attack, score)),
        "pr_auc": float(average_precision_score(is_attack, score)),
        "attack_share": float(is_attack.mean()),
        "recall_at_1pct_fpr_oracle": oracle_recall,
        "oracle_threshold": oracle_t,
        "per_category": per_cat,
    }


def in_domain_reference(unsw: pl.DataFrame, spec: FeatureSpec, seed: int) -> dict[str, Any]:
    """Binary LightGBM trained and tested on UNSW-NB15 itself (temporal split per label)."""
    df = temporal_split(unsw, 0.6, 0.2, gap_rows=50, group_cols=("label",))
    parts = {s: df.filter(pl.col("split") == s) for s in ("train", "val", "test")}
    X = {s: spec.to_numpy(p) for s, p in parts.items()}
    y = {s: (p["label"] != "Benign").to_numpy().astype(np.int64) for s, p in parts.items()}
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "verbosity": -1,
        "metric": "auc",
    }
    booster = lgb.train(
        params,
        lgb.Dataset(X["train"], y["train"]),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(X["val"], y["val"])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    score = np.asarray(booster.predict(X["test"], num_iteration=booster.best_iteration))
    val_score = np.asarray(booster.predict(X["val"], num_iteration=booster.best_iteration))
    # Threshold chosen on UNSW validation at 1% FPR, then applied to UNSW test.
    t = float(np.nextafter(np.quantile(val_score[y["val"] == 0], 1 - TARGET_FPR), np.inf))
    is_attack = y["test"] == 1
    flagged = score >= t
    labels = parts["test"]["label"].to_numpy()
    return {
        "train_flows": len(y["train"]),
        "test_flows": len(y["test"]),
        "roc_auc": float(roc_auc_score(is_attack, score)),
        "pr_auc": float(average_precision_score(is_attack, score)),
        "benign_fpr": float(flagged[~is_attack].mean()),
        "attack_recall": float(flagged[is_attack].mean()),
        "per_category_flagged": {
            c: float(flagged[labels == c].mean()) for c in sorted(set(labels)) if c != "Benign"
        },
    }


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index with quantile bins of `expected`. >0.25 = major shift.

    Low-cardinality features (flags, counts that are mostly zero) get one bin per observed
    value plus open bins below and above, so a shift to a value never seen in `expected`
    still registers.
    """
    vals = np.unique(expected)
    if len(vals) <= bins:
        delta = 1e-9 * (1 + np.abs(vals))
        mids = (vals[:-1] + vals[1:]) / 2
        edges = np.concatenate(
            [[-np.inf, vals[0] - delta[0]], mids, [vals[-1] + delta[-1], np.inf]]
        )
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_shift(
    cic_benign: np.ndarray,
    unsw_benign: np.ndarray,
    names: list[str],
    importance: np.ndarray,
    raw_cic: np.ndarray,
    raw_unsw: np.ndarray,
    top: int = 15,
) -> list[dict[str, Any]]:
    """PSI of the most important features between the two networks' benign traffic."""
    order = np.argsort(-importance)[:top]
    return [
        {
            "feature": names[i],
            "gain_share": float(importance[i] / importance.sum()),
            "psi_benign": psi(cic_benign[:, i], unsw_benign[:, i]),
            "cic_benign_median": float(np.median(raw_cic[:, i])),
            "unsw_benign_median": float(np.median(raw_unsw[:, i])),
        }
        for i in order
    ]


def run(params: Params, csv_path: Path) -> dict[str, Any]:
    ids = params.ids
    target = ids.benign_fpr_target
    processed = params.resolve(params.data.processed_dir)
    full_spec = FeatureSpec.load(processed / "feature_spec.json")
    unsw = load_unsw(csv_path, params.resolve(params.data.interim_dir) / "cic_unsw_nb15.parquet")
    spec = shared_spec(full_spec, set(add_derived(unsw.head(0)).columns))
    dropped = sorted(set(full_spec.columns) - set(spec.columns))
    s = splits_from_frame(pl.read_parquet(processed / "flows.parquet"), spec)
    X_unsw = spec.to_numpy(unsw)
    log.info(
        "CIC train %d flows; UNSW %d flows, %d features",
        len(s.train.y),
        len(unsw),
        len(spec.columns),
    )

    def power(name: str) -> float:
        override = getattr(ids, name).class_weight_power
        return ids.class_weight_power if override is None else override

    cap = ids.max_class_weight
    fits: dict[str, Any] = {
        "lightgbm": lambda: fit_lightgbm(
            s.train.X,
            s.train.y,
            s.val.X,
            s.val.y,
            ids.lightgbm,
            sample_weights(s.train.y, power("lightgbm"), cap),
            params.seed,
            s.feature_names,
        ),
        "xgboost": lambda: fit_xgboost(
            s.train.X,
            s.train.y,
            s.val.X,
            s.val.y,
            ids.xgboost,
            sample_weights(s.train.y, power("xgboost"), cap),
            params.seed,
        ),
        "mlp": lambda: fit_mlp(
            s.train.X,
            s.train.y,
            s.val.X,
            s.val.y,
            ids.mlp,
            class_weights(s.train.y, ids.class_weight_power, cap),
            params.seed,
        ),
    }

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("sentinel-ids-crossdataset")
    results: dict[str, Any] = {"n_features": len(spec.columns), "dropped": dropped}
    scored: dict[str, Scored] = {}
    unsw_proba: dict[str, np.ndarray] = {}
    with mlflow.start_run(run_name="cic2017-to-unsw-nb15"):
        for name, fit in fits.items():
            t0 = time.perf_counter()
            model: LogitModel = fit()
            sc = score_model(model, s, target, time.perf_counter() - t0)
            scored[name] = sc
            unsw_proba[name] = sc.calibration.apply(model.logits(X_unsw))
            results[name] = {
                "cic_test_macro_f1": sc.test_metrics["macro_f1"],
                "cic_test_benign_fpr": sc.test_metrics["benign_fpr"],
                "unsw": transfer_metrics(unsw_proba[name], sc.threshold, unsw["label"]),
            }
            log.info(
                "%s: CIC macro-F1 %.4f | UNSW %s",
                name,
                sc.test_metrics["macro_f1"],
                {k: round(v, 4) for k, v in results[name]["unsw"].items() if isinstance(v, float)},
            )

        lg, mlp = scored["lightgbm"], scored["mlp"]
        w, t = choose_ensemble_weight(lg.val_proba, mlp.val_proba, s.val.y, target)
        ens_cic = w * lg.test_proba + (1 - w) * mlp.test_proba
        ens_unsw = w * unsw_proba["lightgbm"] + (1 - w) * unsw_proba["mlp"]
        cic_m = evaluate(s.test.y, decide(ens_cic, t), ens_cic)
        results["ensemble"] = {
            "lgbm_weight": w,
            "cic_test_macro_f1": cic_m["macro_f1"],
            "cic_test_benign_fpr": cic_m["benign_fpr"],
            "unsw": transfer_metrics(ens_unsw, t, unsw["label"]),
        }
        results["unsw_in_domain_reference"] = in_domain_reference(unsw, spec, params.seed)

        assert isinstance(lg.model, LightGBMModel)
        gain = lg.model.booster.feature_importance("gain").astype(np.float64)
        cic_b = s.train.X[s.train.y == BENIGN]
        unsw_b_mask = (unsw["label"] == "Benign").to_numpy()
        rng = np.random.default_rng(params.seed)
        cic_idx = rng.choice(len(cic_b), size=min(200_000, len(cic_b)), replace=False)
        cic_frame = s.train.frame.filter(pl.col("family") == "Benign")[cic_idx.tolist()]
        results["feature_shift"] = feature_shift(
            cic_b[cic_idx],
            X_unsw[unsw_b_mask],
            spec.columns,
            gain,
            spec.raw(cic_frame),
            spec.raw(unsw.filter(pl.col("label") == "Benign")),
        )

        for name in ("lightgbm", "xgboost", "mlp", "ensemble"):
            u = results[name]["unsw"]
            mlflow.log_metrics({f"{name}_{k}": v for k, v in u.items() if isinstance(v, float)})
            mlflow.log_metric(f"{name}_cic_test_macro_f1", results[name]["cic_test_macro_f1"])
        ref = results["unsw_in_domain_reference"]
        mlflow.log_metrics({f"reference_{k}": v for k, v in ref.items() if isinstance(v, float)})

        reports = params.resolve(params.data.reports_dir) / "ids"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "cross_dataset.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (reports / "cross_dataset.md").write_text(report(results), encoding="utf-8")
        (reports / "cross_dataset_metrics.json").write_text(
            json.dumps(
                {
                    n: {
                        k: results[n]["unsw"][k] for k in ("roc_auc", "attack_recall", "benign_fpr")
                    }
                    for n in ("lightgbm", "xgboost", "mlp", "ensemble")
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(reports / "cross_dataset.md"))
    return results


def report(r: dict[str, Any]) -> str:
    models = ("lightgbm", "xgboost", "mlp", "ensemble")
    ref = r["unsw_in_domain_reference"]
    lines = [
        "# Cross-dataset test: CIC-IDS2017 -> UNSW-NB15\n",
        f"Models trained on CIC-IDS2017 ({r['n_features']} shared CICFlowMeter features; "
        f"dropped {', '.join(r['dropped'])}) and applied unchanged to CIC-UNSW-NB15. "
        "Threshold and calibration come from CIC-IDS2017 validation.\n",
        "## Benign vs attack\n",
        "| Model | CIC test macro-F1 | UNSW benign FPR | UNSW attack recall | UNSW precision "
        "| UNSW ROC-AUC | UNSW PR-AUC | Recall @1% FPR (UNSW-chosen threshold) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in models:
        u = r[n]["unsw"]
        lines.append(
            f"| {n} | {r[n]['cic_test_macro_f1']:.4f} | {u['benign_fpr']:.2%} | "
            f"{u['attack_recall']:.2%} | {u['precision']:.2%} | {u['roc_auc']:.4f} | "
            f"{u['pr_auc']:.4f} | {u['recall_at_1pct_fpr_oracle']:.2%} |"
        )
    lines.append(
        f"| *UNSW-trained reference* | - | {ref['benign_fpr']:.2%} | {ref['attack_recall']:.2%} "
        f"| - | {ref['roc_auc']:.4f} | {ref['pr_auc']:.4f} | - |"
    )
    ens = r["ensemble"]["unsw"]["per_category"]
    lines += [
        "\n## Ensemble detection rate per UNSW-NB15 category\n",
        "| UNSW category | Flows | Flagged as attack (CIC-trained) "
        "| Flagged (UNSW-trained reference) "
        "| Most common predicted family |",
        "|---|---|---|---|---|",
    ]
    for cat, e in sorted(ens.items(), key=lambda kv: -kv[1]["flows"]):
        top = (
            max(e["predicted_as"].items(), key=lambda kv: kv[1])[0] if "predicted_as" in e else "-"
        )
        refv = ref["per_category_flagged"].get(cat)
        refs = f"{refv:.2%}" if refv is not None else "-"
        extra = "".join(
            f" ({v:.0%} of detections {k})" for k, v in e.items() if k.startswith("as_")
        )
        lines.append(f"| {cat} | {e['flows']:,} | {e['flagged']:.2%}{extra} | {refs} | {top} |")
    lines += [
        "\n## Where the shift comes from (benign traffic, top LightGBM features)\n",
        "PSI above 0.25 is conventionally a major distribution shift.\n",
        "| Feature | Share of gain | PSI (benign CIC vs UNSW) | Median CIC | Median UNSW |",
        "|---|---|---|---|---|",
    ]
    for f in r["feature_shift"]:
        lines.append(
            f"| `{f['feature']}` | {f['gain_share']:.1%} | {f['psi_benign']:.2f} | "
            f"{f['cic_benign_median']:,.4g} | {f['unsw_benign_median']:,.4g} |"
        )
    return "\n".join(lines) + "\n"
