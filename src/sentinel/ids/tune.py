"""Optuna search for LightGBM and XGBoost with the same trial budget.

Tune on the full training split (the default). The dev sample keeps rare labels whole
but only 10% of benign traffic, so rare classes are ~10x more prevalent there than in
reality. Precision depends on prevalence, so dev-tuned settings that overfit the ~20
Infiltration flows look fine on dev and produce ~100 benign false positives per rare
class on the real mix (see README, Module 1). `--sample dev` remains for quick checks.

The objective is the deployed decision rule's validation macro-F1: calibrate on
validation, pick the benign-FPR-constrained threshold, score. The test split is never
seen here. The class-weight power is searched too, since it interacts with tree capacity.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import mlflow
import optuna

from sentinel.common.config import LightGBMParams, Params, XGBoostParams, get_settings
from sentinel.common.logging import get_logger
from sentinel.ids.calibration import TemperatureBias
from sentinel.ids.dataset import Sample, load_splits
from sentinel.ids.decision import choose_threshold
from sentinel.ids.models import LogitModel
from sentinel.ids.models.gbm import fit_lightgbm, fit_xgboost
from sentinel.ids.weights import sample_weights

log = get_logger(__name__)

Family = Literal["lightgbm", "xgboost"]


def _lightgbm_space(trial: optuna.Trial) -> LightGBMParams:
    return LightGBMParams(
        n_estimators=1000,
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
        early_stopping_rounds=50,
    )


def _xgboost_space(trial: optuna.Trial) -> XGBoostParams:
    return XGBoostParams(
        n_estimators=1000,
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_float("min_child_weight", 0.1, 20, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        early_stopping_rounds=50,
    )


def tune(
    params: Params,
    family: Family,
    n_trials: int | None = None,
    sample: Sample = "full",
    timeout_minutes: float | None = None,
) -> dict[str, Any]:
    """Budget: `n_trials` or `timeout_minutes`, whichever comes first. A full-data XGBoost
    fit is ~10x slower than LightGBM, so an equal time budget keeps the comparison fair."""
    n_trials = n_trials or params.ids.tune_trials
    timeout_minutes = timeout_minutes or params.ids.tune_timeout_minutes
    s = load_splits(params, sample)
    target = params.ids.benign_fpr_target
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("sentinel-ids-tuning")

    def objective(trial: optuna.Trial) -> float:
        power = trial.suggest_float("class_weight_power", 0.0, 1.0)
        sw = sample_weights(s.train.y, power, params.ids.max_class_weight)
        with mlflow.start_run(run_name=f"{family}-{sample}-trial-{trial.number}", nested=True):
            model: LogitModel
            if family == "lightgbm":
                lp = _lightgbm_space(trial)
                lm = fit_lightgbm(s.train.X, s.train.y, s.val.X, s.val.y, lp, sw, params.seed)
                model, rounds, p = lm, lm.booster.best_iteration, lp.model_dump()
            else:
                xp = _xgboost_space(trial)
                xm = fit_xgboost(s.train.X, s.train.y, s.val.X, s.val.y, xp, sw, params.seed)
                model, rounds, p = xm, int(xm.model.best_iteration) + 1, xp.model_dump()
            logits = model.logits(s.val.X)
            proba = TemperatureBias.fit(logits, s.val.y).apply(logits)
            f1 = choose_threshold(proba, s.val.y, target).val_macro_f1
            trial.set_user_attr("rounds", rounds)
            mlflow.log_params({**p, "class_weight_power": power})
            mlflow.log_metrics({"val_macro_f1": f1, "rounds": rounds})
            return f1

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mlflow.start_run(run_name=f"tune-{family}-{sample}"):
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=params.seed)
        )
        # Seed the search with the params.yaml settings and the library defaults, so the
        # result is at least as good as either on validation.
        current = params.ids.lightgbm if family == "lightgbm" else params.ids.xgboost
        default = LightGBMParams() if family == "lightgbm" else XGBoostParams()
        log_floor = ("lambda_l1", "lambda_l2", "reg_alpha", "reg_lambda")
        for seed_params in (current, default):
            start = {
                k: max(v, 1e-8) if k in log_floor else v
                for k, v in seed_params.model_dump().items()
                if k not in ("n_estimators", "early_stopping_rounds")
            }
            study.enqueue_trial({**start, "class_weight_power": params.ids.class_weight_power})
        study.optimize(objective, n_trials=n_trials, timeout=timeout_minutes * 60)
        best = {**study.best_params, "rounds": study.best_trial.user_attrs["rounds"]}
        mlflow.log_metric("best_val_macro_f1", study.best_value)
        mlflow.log_dict(best, "best_params.json")

    out = {
        "family": family,
        "sample": sample,
        "n_trials": len(study.trials),
        "timeout_minutes": timeout_minutes,
        "best_val_macro_f1": study.best_value,
        "best_params": best,
    }
    reports = params.resolve(params.data.reports_dir) / "ids"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"tuning_{family}_{sample}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    log.info("best %s: %.4f %s", family, study.best_value, best)
    return out
