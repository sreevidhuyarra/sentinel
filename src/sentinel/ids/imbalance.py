"""Imbalance study: which remedy works best for rare attack families?

Runs on the dev sample. Every strategy goes through the same calibrate-on-val and
benign-FPR-constrained threshold, so differences reflect the training remedy rather
than how the decision threshold happens to fall. Resampling touches the training fold
only; validation and test keep the real class mix.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

import mlflow
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sentinel.common.config import LightGBMParams, Params, get_settings
from sentinel.common.logging import get_logger
from sentinel.ids.dataset import BENIGN, CLASSES, Split, Splits, load_splits
from sentinel.ids.models.gbm import fit_lightgbm
from sentinel.ids.models.mlp import fit_mlp
from sentinel.ids.train import score_model
from sentinel.ids.weights import class_weights, sample_weights

log = get_logger(__name__)

RARE = ["Infiltration", "WebAttack", "Bot"]
_QUICK_GBM = LightGBMParams(n_estimators=600, learning_rate=0.1, early_stopping_rounds=30)
OVERSAMPLE_TO = 0.02  # minority classes raised to 2% of the benign count


def _oversample_targets(y: np.ndarray) -> dict[int, int]:
    counts = np.bincount(y, minlength=len(CLASSES))
    target = int(OVERSAMPLE_TO * counts[BENIGN])
    return {c: target for c in range(len(CLASSES)) if 0 < counts[c] < target}


def _resampled(s: Splits, X: np.ndarray, y: np.ndarray) -> Splits:
    return replace(s, train=Split(X.astype(np.float32), y.astype(np.int64), s.train.frame))


def study(params: Params) -> dict[str, Any]:
    s = load_splits(params, "dev")
    target = params.ids.benign_fpr_target
    seed = params.seed
    y = s.train.y

    def gbm(split: Splits, weight: np.ndarray | None) -> Any:
        return fit_lightgbm(
            split.train.X, split.train.y, split.val.X, split.val.y, _QUICK_GBM, weight, seed
        )

    def mlp(gamma: float) -> Any:
        quick = params.ids.mlp.model_copy(
            update={"epochs": 12, "patience": 3, "focal_gamma": gamma}
        )
        cw = class_weights(y, 0.5, cap)
        return fit_mlp(s.train.X, y, s.val.X, s.val.y, quick, cw, seed)

    cap = params.ids.max_class_weight
    counts = np.bincount(y, minlength=len(CLASSES))
    k = int(max(1, min(5, int(counts[counts > 0].min()) - 1)))
    targets = _oversample_targets(y)
    smote = SMOTE(sampling_strategy=targets, k_neighbors=k, random_state=seed)
    ros = RandomOverSampler(sampling_strategy=targets, random_state=seed)
    smote_split = _resampled(s, *smote.fit_resample(s.train.X, y))
    ros_split = _resampled(s, *ros.fit_resample(s.train.X, y))

    strategies: dict[str, Any] = {
        "LightGBM, unweighted": lambda: (gbm(s, None), s),
        "LightGBM, sqrt class weights": lambda: (gbm(s, sample_weights(y, 0.5, cap)), s),
        "LightGBM, inverse-frequency weights": lambda: (gbm(s, sample_weights(y, 1.0, cap)), s),
        "LightGBM, random oversampling": lambda: (gbm(ros_split, None), ros_split),
        "LightGBM, SMOTE": lambda: (gbm(smote_split, None), smote_split),
        "MLP, cross-entropy + sqrt weights": lambda: (mlp(0.0), s),
        "MLP, focal loss (gamma=2) + sqrt weights": lambda: (mlp(2.0), s),
    }

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("sentinel-ids-imbalance")
    rows: dict[str, dict[str, Any]] = {}
    with mlflow.start_run(run_name="imbalance-study"):
        for name, run in strategies.items():
            with mlflow.start_run(run_name=name, nested=True):
                t0 = time.perf_counter()
                model, split = run()
                secs = time.perf_counter() - t0
                m = score_model(model, split, target, secs).test_metrics
                rows[name] = {
                    "val_macro_f1": m["val_macro_f1"],
                    "test_macro_f1": m["macro_f1"],
                    "test_benign_fpr": m["benign_fpr"],
                    **{f"recall_{c}": m["per_class"][c]["recall"] for c in RARE},
                    "train_rows": len(split.train.y),
                    "seconds": secs,
                }
                mlflow.log_metrics({k: float(v) for k, v in rows[name].items()})
                log.info("%s: %s", name, rows[name])

    reports = params.resolve(params.data.reports_dir) / "ids"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "imbalance.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (reports / "imbalance.md").write_text(_table(rows), encoding="utf-8")
    return rows


def _table(rows: dict[str, dict[str, Any]]) -> str:
    head = (
        "# Imbalance study (dev sample)\n\nEach strategy is calibrated and thresholded on "
        "validation (benign FPR <= 1%), then scored on test.\n\n"
        "| Strategy | Val macro-F1 | Test macro-F1 | Test benign FPR | "
        + " | ".join(f"Recall {c}" for c in RARE)
        + " | Train rows | Seconds |\n|"
        + "---|" * (6 + len(RARE))
        + "\n"
    )
    body = ""
    for name, r in rows.items():
        rare = " | ".join(f"{r[f'recall_{c}']:.3f}" for c in RARE)
        body += (
            f"| {name} | {r['val_macro_f1']:.4f} | {r['test_macro_f1']:.4f} | "
            f"{r['test_benign_fpr']:.3%} | {rare} | {r['train_rows']:,} | {r['seconds']:.0f} |\n"
        )
    return head + body
