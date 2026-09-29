"""Train and evaluate the anomaly detector (Module 2).

Protocol:
  benign train flows       -> fit autoencoders (one per bottleneck size) and Isolation Forest
  validation               -> early stopping (benign MSE), pick the bottleneck (ROC-AUC
                              benign vs known attacks), set the threshold (benign FPR target)
  test                     -> report; the test split is not used for any choice
Then three studies:
  fusion   - deployed supervised model + autoencoder on the normal test split
  holdout  - for each family in split.holdout_families, retrain LightGBM without it and
             measure how much of it the supervised model, the anomaly detectors and the
             fused detector catch (the unknown-attack case the module exists for)
  unsw     - on CIC-UNSW-NB15, compare the CIC-trained autoencoder as is, re-thresholded
             on the new network's benign traffic, and retrained on it (no attack labels)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl

from sentinel.anomaly.autoencoder import AutoencoderModel, fit_autoencoder
from sentinel.anomaly.bundle import AnomalyBundle
from sentinel.anomaly.evaluate import detection, recall_at_fprs, threshold_at_fpr
from sentinel.anomaly.iforest import fit_iforest
from sentinel.anomaly.registry import log_bundle, promote
from sentinel.anomaly.report import model_card, plot_scores, results_markdown
from sentinel.common.config import Params, get_settings
from sentinel.common.logging import get_logger
from sentinel.data.features import FeatureSpec, add_derived
from sentinel.data.splits import temporal_split
from sentinel.ids.bundle import IDSBundle
from sentinel.ids.dataset import BENIGN, CLASSES, Splits, splits_from_frame
from sentinel.ids.decision import decide
from sentinel.ids.models.gbm import fit_lightgbm
from sentinel.ids.train import score_model
from sentinel.ids.weights import sample_weights

log = get_logger(__name__)

EXPERIMENT = "sentinel-anomaly"


def _families(split: Any) -> np.ndarray:
    return np.asarray(split.frame["family"].to_numpy())


def _benign(split: Any) -> np.ndarray:
    return np.asarray(split.X[split.y == BENIGN])


def sweep_bottleneck(params: Params, s: Splits) -> tuple[AutoencoderModel, list[dict[str, Any]]]:
    ap = params.anomaly.autoencoder
    val_labels = _families(s.val)
    rows, best, best_auc = [], None, -1.0
    for k in ap.bottleneck_sweep:
        with mlflow.start_run(run_name=f"autoencoder-k{k}", nested=True):
            t0 = time.perf_counter()
            model = fit_autoencoder(_benign(s.train), _benign(s.val), ap, k, params.seed)
            secs = time.perf_counter() - t0
            d = detection(model.score(s.val.X), np.inf, val_labels)
            row = {
                "bottleneck": k,
                "val_roc_auc": d["roc_auc"],
                "epochs": len(model.history),
                "val_benign_mse": model.history[-1]["val_mse"],
                "seconds": secs,
            }
            mlflow.log_params({"bottleneck": k, **ap.model_dump(exclude={"bottleneck_sweep"})})
            mlflow.log_metrics({k2: float(v) for k2, v in row.items()})
            rows.append(row)
            log.info("autoencoder k=%d: val ROC-AUC %.4f", k, d["roc_auc"])
            if d["roc_auc"] > best_auc:
                best, best_auc = model, d["roc_auc"]
    assert best is not None
    return best, rows


def evaluate_scores(
    name: str,
    val_score: np.ndarray,
    val_labels: np.ndarray,
    test_score: np.ndarray,
    test_labels: np.ndarray,
    fpr: float,
    fprs: list[float],
) -> dict[str, Any]:
    benign_val = val_score[val_labels == "Benign"]
    t = threshold_at_fpr(benign_val, fpr)
    return {
        "name": name,
        "threshold": t,
        "test": detection(test_score, t, test_labels),
        "fpr_sweep": recall_at_fprs(benign_val, test_score, test_labels, fprs),
    }


def fusion_study(
    ids: IDSBundle, s: Splits, ae_test: np.ndarray, threshold: float
) -> dict[str, Any]:
    labels = _families(s.test)
    sup_pred = decide(ids.proba(s.test.X), ids.threshold)
    sup_flag = sup_pred != BENIGN
    fused = sup_flag | (ae_test >= threshold)
    benign = labels == "Benign"
    out: dict[str, Any] = {}
    for name, flag in (("supervised", sup_flag), ("fused", fused)):
        out[name] = {
            "benign_fpr": float(flag[benign].mean()),
            "benign_false_alarms": int(flag[benign].sum()),
            "attack_recall": float(flag[~benign].mean()),
            "per_family_recall": {
                f: float(flag[labels == f].mean()) for f in CLASSES if f != "Benign"
            },
        }
    out["unknown_anomaly_alerts"] = {
        "on_benign": int((fused & ~sup_flag)[benign].sum()),
        "on_attacks": int((fused & ~sup_flag)[~benign].sum()),
    }
    return out


def holdout_study(
    params: Params,
    frame: pl.DataFrame,
    spec: FeatureSpec,
    ae: AutoencoderModel,
    ae_threshold: float,
    iforest: Any,
    if_threshold: float,
) -> dict[str, Any]:
    ids = params.ids
    power = ids.lightgbm.class_weight_power
    power = ids.class_weight_power if power is None else power
    out: dict[str, Any] = {}
    for family in params.split.holdout_families:
        with mlflow.start_run(run_name=f"holdout-{family}", nested=True):
            moved = frame.with_columns(
                pl.when(pl.col("family") == family)
                .then(pl.lit("test"))
                .otherwise(pl.col("split"))
                .alias("split")
            )
            s = splits_from_frame(moved, spec)
            assert not (s.train.y == CLASSES.index(family)).any()
            sw = sample_weights(s.train.y, power, ids.max_class_weight)
            model = fit_lightgbm(
                s.train.X,
                s.train.y,
                s.val.X,
                s.val.y,
                ids.lightgbm,
                sw,
                params.seed,
                s.feature_names,
            )
            sc = score_model(model, s, ids.benign_fpr_target, 0.0)
            pred = decide(sc.test_proba, sc.threshold)
            labels = _families(s.test)
            target, benign = labels == family, labels == "Benign"
            sup = pred != BENIGN
            ae_flag = ae.score(s.test.X) >= ae_threshold
            if_flag = iforest.score(s.test.X) >= if_threshold
            counts = np.bincount(pred[target], minlength=len(CLASSES))
            res = {
                "flows": int(target.sum()),
                "supervised_recall": float(sup[target].mean()),
                "supervised_predicted_as": {CLASSES[i]: int(n) for i, n in enumerate(counts) if n},
                "autoencoder_recall": float(ae_flag[target].mean()),
                "iforest_recall": float(if_flag[target].mean()),
                "fused_recall": float((sup | ae_flag)[target].mean()),
                "fused_iforest_recall": float((sup | if_flag)[target].mean()),
                "supervised_benign_fpr": float(sup[benign].mean()),
                "fused_benign_fpr": float((sup | ae_flag)[benign].mean()),
                "known_families_macro_f1": sc.test_metrics["macro_f1"],
            }
            mlflow.log_metrics({k: float(v) for k, v in res.items() if isinstance(v, int | float)})
            log.info("holdout %s: %s", family, res)
            out[family] = res
    return out


def unsw_study(
    params: Params,
    s_frame: pl.DataFrame,
    full_spec: FeatureSpec,
    bottleneck: int,
    unsw_parquet: Path,
) -> dict[str, Any]:
    from sentinel.ids.crossdataset import shared_spec

    a = params.anomaly
    unsw = pl.read_parquet(unsw_parquet)
    spec = shared_spec(full_spec, set(add_derived(unsw.head(0)).columns))
    u = temporal_split(unsw, 0.6, 0.2, gap_rows=50, group_cols=("label",))
    parts = {k: u.filter(pl.col("split") == k) for k in ("train", "val", "test")}
    X = {k: spec.to_numpy(p) for k, p in parts.items()}
    lab = {k: np.asarray(p["label"].to_numpy()) for k, p in parts.items()}
    ub = {k: X[k][lab[k] == "Benign"] for k in X}

    cic = splits_from_frame(s_frame, spec)
    cb_train, cb_val = _benign(cic.train), _benign(cic.val)
    ae_cic = fit_autoencoder(cb_train, cb_val, a.autoencoder, bottleneck, params.seed)
    ae_unsw = fit_autoencoder(ub["train"], ub["val"], a.autoencoder, bottleneck, params.seed)
    if_unsw = fit_iforest(ub["train"], a.iforest, params.seed)

    cic_score_test = ae_cic.score(X["test"])
    t_cic = threshold_at_fpr(ae_cic.score(cb_val), a.fpr_target)
    t_rethreshold = threshold_at_fpr(ae_cic.score(ub["val"]), a.fpr_target)
    unsw_score_test = ae_unsw.score(X["test"])
    t_unsw = threshold_at_fpr(ae_unsw.score(ub["val"]), a.fpr_target)
    if_test = if_unsw.score(X["test"])
    t_if = threshold_at_fpr(if_unsw.score(ub["val"]), a.fpr_target)
    return {
        "n_features": len(spec.columns),
        "test_flows": len(lab["test"]),
        "cic_autoencoder_as_is": detection(cic_score_test, t_cic, lab["test"]),
        "cic_autoencoder_rethresholded": detection(cic_score_test, t_rethreshold, lab["test"]),
        "unsw_autoencoder_benign_only": detection(unsw_score_test, t_unsw, lab["test"]),
        "unsw_iforest_benign_only": detection(if_test, t_if, lab["test"]),
    }


def train(
    params: Params,
    register: bool = True,
    out_dir: Path | None = None,
    reports_dir: Path | None = None,
    ids_bundle_dir: Path | None = None,
    studies: tuple[str, ...] = ("fusion", "holdout", "unsw"),
) -> dict[str, Any]:
    a = params.anomaly
    out_dir = out_dir or params.resolve(Path("models/anomaly"))
    reports_dir = reports_dir or params.resolve(params.data.reports_dir) / "anomaly"
    ids_bundle_dir = ids_bundle_dir or params.resolve(Path("models/ids"))
    processed = params.resolve(params.data.processed_dir)
    frame = pl.read_parquet(processed / "flows.parquet")
    spec = FeatureSpec.load(processed / "feature_spec.json")
    s = splits_from_frame(frame, spec)
    val_labels, test_labels = _families(s.val), _families(s.test)

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    results: dict[str, Any] = {}
    with mlflow.start_run(run_name="anomaly-train") as parent:
        mlflow.log_params(
            {
                "sample": "full",
                "fpr_target": a.fpr_target,
                "benign_train_flows": int((s.train.y == BENIGN).sum()),
            }
        )
        ae, sweep = sweep_bottleneck(params, s)
        results["bottleneck_sweep"] = sweep
        results["bottleneck"] = ae.bottleneck

        t0 = time.perf_counter()
        forest = fit_iforest(_benign(s.train), a.iforest, params.seed)
        if_secs = time.perf_counter() - t0

        ae_val, ae_test = ae.score(s.val.X), ae.score(s.test.X)
        if_val, if_test = forest.score(s.val.X), forest.score(s.test.X)
        results["autoencoder"] = evaluate_scores(
            "autoencoder", ae_val, val_labels, ae_test, test_labels, a.fpr_target, a.fpr_sweep
        )
        results["iforest"] = evaluate_scores(
            "iforest", if_val, val_labels, if_test, test_labels, a.fpr_target, a.fpr_sweep
        )
        results["iforest"]["train_seconds"] = if_secs
        ae_t, if_t = results["autoencoder"]["threshold"], results["iforest"]["threshold"]
        for name in ("autoencoder", "iforest"):
            t = results[name]["test"]
            mlflow.log_metrics(
                {f"{name}_test_{k}": float(v) for k, v in t.items() if isinstance(v, float)}
            )
        mlflow.log_metrics(
            {
                "test_roc_auc": results["autoencoder"]["test"]["roc_auc"],
                "test_benign_fpr": results["autoencoder"]["test"]["benign_fpr"],
            }
        )

        if "fusion" in studies and (ids_bundle_dir / "bundle.json").exists():
            results["fusion"] = fusion_study(IDSBundle.load(ids_bundle_dir), s, ae_test, ae_t)
        if "holdout" in studies:
            results["holdout"] = holdout_study(params, frame, spec, ae, ae_t, forest, if_t)
        unsw_parquet = params.resolve(params.data.interim_dir) / "cic_unsw_nb15.parquet"
        if "unsw" in studies and unsw_parquet.exists():
            results["unsw"] = unsw_study(params, frame, spec, ae.bottleneck, unsw_parquet)

        bundle = AnomalyBundle(
            spec,
            ae,
            ae_t,
            a.fpr_target,
            metadata={"parent_run_id": parent.info.run_id, "bottleneck": ae.bottleneck},
        )
        bundle.save(out_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        plot_scores(ae_test, test_labels, ae_t, reports_dir / "figures" / "autoencoder_scores.png")
        (reports_dir / "results.json").write_text(
            json.dumps(results, indent=2, default=float), encoding="utf-8"
        )
        (reports_dir / "results.md").write_text(results_markdown(results), encoding="utf-8")
        (reports_dir / "model_card.md").write_text(
            model_card(results, a.fpr_target), encoding="utf-8"
        )
        metrics = {
            n: {
                "roc_auc": results[n]["test"]["roc_auc"],
                "attack_recall": results[n]["test"]["attack_recall"],
                "benign_fpr": results[n]["test"]["benign_fpr"],
            }
            for n in ("autoencoder", "iforest")
        }
        (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
        version = log_bundle(out_dir, register=register)
        decision = promote(version, 1.5 * a.fpr_target) if version else None
    results["registry"] = decision
    return results
