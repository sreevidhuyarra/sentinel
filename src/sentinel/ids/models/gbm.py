"""LightGBM (production classifier) and XGBoost (comparison) with macro-F1 early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from sentinel.common.config import LightGBMParams, XGBoostParams
from sentinel.ids.dataset import CLASSES
from sentinel.ids.metrics import macro_f1


@dataclass
class LightGBMModel:
    booster: lgb.Booster

    def logits(self, X: np.ndarray) -> np.ndarray:
        out = self.booster.predict(X, raw_score=True, num_iteration=self.booster.best_iteration)
        return np.asarray(out, dtype=np.float64)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Exact TreeSHAP values, shape (n, n_classes, n_features + 1); last column is the
        expected value. Same algorithm as shap.TreeExplainer, computed natively."""
        raw = self.booster.predict(X, pred_contrib=True, num_iteration=self.booster.best_iteration)
        n, k = X.shape[0], len(CLASSES)
        return np.asarray(raw).reshape(n, k, X.shape[1] + 1)

    def save(self, path: Path) -> None:
        self.booster.save_model(str(path), num_iteration=self.booster.best_iteration)

    @classmethod
    def load(cls, path: Path) -> LightGBMModel:
        return cls(lgb.Booster(model_file=str(path)))


def lightgbm_params(p: LightGBMParams, seed: int, n_jobs: int = -1) -> dict[str, Any]:
    return {
        "objective": "multiclass",
        "num_class": len(CLASSES),
        "metric": "None",  # early stopping uses only the macro-F1 feval
        "learning_rate": p.learning_rate,
        "num_leaves": p.num_leaves,
        "min_child_samples": p.min_child_samples,
        "feature_fraction": p.feature_fraction,
        "bagging_fraction": p.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l1": p.lambda_l1,
        "lambda_l2": p.lambda_l2,
        "seed": seed,
        "num_threads": n_jobs,
        "verbosity": -1,
    }


def fit_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    p: LightGBMParams,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
    feature_names: list[str] | None = None,
) -> LightGBMModel:
    def feval(preds: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
        return "macro_f1", macro_f1(y_val, preds.argmax(axis=1)), True

    names: list[str] | Literal["auto"] = feature_names or "auto"
    train = lgb.Dataset(X, y, weight=sample_weight, feature_name=names, free_raw_data=True)
    # Validation stays unweighted: it should measure performance on real traffic.
    val = lgb.Dataset(X_val, y_val, reference=train, feature_name=names)
    booster = lgb.train(
        lightgbm_params(p, seed),
        train,
        num_boost_round=p.n_estimators,
        valid_sets=[val],
        valid_names=["val"],
        feval=feval,
        callbacks=[lgb.early_stopping(p.early_stopping_rounds, verbose=False)],
    )
    return LightGBMModel(booster)


@dataclass
class XGBoostModel:
    model: xgb.XGBClassifier

    def logits(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X, output_margin=True), dtype=np.float64)


def fit_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    p: XGBoostParams,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
) -> XGBoostModel:
    def one_minus_macro_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
        return 1.0 - macro_f1(y_true.astype(np.int64), proba.argmax(axis=1))

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        tree_method="hist",
        n_estimators=p.n_estimators,
        learning_rate=p.learning_rate,
        max_depth=p.max_depth,
        min_child_weight=p.min_child_weight,
        subsample=p.subsample,
        colsample_bytree=p.colsample_bytree,
        reg_alpha=p.reg_alpha,
        reg_lambda=p.reg_lambda,
        early_stopping_rounds=p.early_stopping_rounds,
        eval_metric=one_minus_macro_f1,
        disable_default_eval_metric=True,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
    return XGBoostModel(model)
