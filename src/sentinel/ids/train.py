"""Train, calibrate, threshold, evaluate and register the IDS models.

Protocol (the test split is touched exactly once per model, at the end):
  train split      -> fit model (early stopping on validation macro-F1)
  validation split -> fit calibration, choose the benign-FPR-constrained threshold,
                      choose the LightGBM/MLP ensemble weight
  test split       -> report every metric
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np

from sentinel.common.config import Params, get_settings
from sentinel.common.logging import get_logger
from sentinel.ids.bundle import IDSBundle
from sentinel.ids.calibration import TemperatureBias
from sentinel.ids.dataset import CLASSES, Sample, Splits, load_splits
from sentinel.ids.decision import choose_threshold, decide
from sentinel.ids.explain import global_importance, plot_global_importance
from sentinel.ids.metrics import evaluate, flat
from sentinel.ids.models import LogitModel
from sentinel.ids.models.gbm import LightGBMModel, fit_lightgbm, fit_xgboost
from sentinel.ids.models.linear import fit_logreg
from sentinel.ids.models.mlp import MLPModel, fit_mlp
from sentinel.ids.registry import log_bundle, promote
from sentinel.ids.report import comparison_table, model_card, per_class_table, plot_confusion
from sentinel.ids.weights import class_weights, sample_weights

log = get_logger(__name__)

ALL_MODELS = ("logreg", "lightgbm", "xgboost", "mlp")
EXPERIMENT = "sentinel-ids"
SHAP_SAMPLE = 20_000


@dataclass
class Scored:
    model: LogitModel
    calibration: TemperatureBias
    threshold: float
    val_proba: np.ndarray
    test_proba: np.ndarray
    test_metrics: dict[str, Any]
    train_seconds: float
    latency_ms_per_1k: float


def latency_ms_per_1k(model: LogitModel, cal: TemperatureBias, X: np.ndarray) -> float:
    batch = X[:1000]
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        cal.apply(model.logits(batch))
        best = min(best, time.perf_counter() - t0)
    return float(best * 1000 * 1000 / len(batch))


def score_model(model: LogitModel, s: Splits, fpr_target: float, train_seconds: float) -> Scored:
    val_logits = model.logits(s.val.X)
    cal = TemperatureBias.fit(val_logits, s.val.y)
    val_proba = cal.apply(val_logits)
    choice = choose_threshold(val_proba, s.val.y, fpr_target)
    test_proba = cal.apply(model.logits(s.test.X))
    metrics = evaluate(s.test.y, decide(test_proba, choice.threshold), test_proba)
    metrics["val_macro_f1"] = choice.val_macro_f1
    metrics["val_benign_fpr"] = choice.val_benign_fpr
    return Scored(
        model,
        cal,
        choice.threshold,
        val_proba,
        test_proba,
        metrics,
        train_seconds,
        latency_ms_per_1k(model, cal, s.test.X),
    )


def choose_ensemble_weight(
    p_lgbm: np.ndarray, p_mlp: np.ndarray, y: np.ndarray, fpr_target: float
) -> tuple[float, float]:
    """LightGBM weight in [0, 1] (step 0.1) with the best constrained validation macro-F1."""
    best_w, best_t, best_f1 = 1.0, 0.5, -1.0
    for w in np.linspace(0, 1, 11):
        choice = choose_threshold(w * p_lgbm + (1 - w) * p_mlp, y, fpr_target)
        if choice.val_macro_f1 > best_f1 + 1e-6:
            best_w, best_t, best_f1 = float(w), choice.threshold, choice.val_macro_f1
    return best_w, best_t


def _fit(name: str, s: Splits, params: Params) -> tuple[LogitModel, float]:
    ids = params.ids
    power = ids.class_weight_power
    if name in ("lightgbm", "xgboost"):
        override = getattr(ids, name).class_weight_power
        power = power if override is None else override
    sw = sample_weights(s.train.y, power, ids.max_class_weight)
    t0 = time.perf_counter()
    model: LogitModel
    if name == "logreg":
        model = fit_logreg(s.train.X, s.train.y, sw, params.seed)
    elif name == "lightgbm":
        model = fit_lightgbm(
            s.train.X, s.train.y, s.val.X, s.val.y, ids.lightgbm, sw, params.seed, s.feature_names
        )
    elif name == "xgboost":
        model = fit_xgboost(s.train.X, s.train.y, s.val.X, s.val.y, ids.xgboost, sw, params.seed)
    elif name == "mlp":
        cw = class_weights(s.train.y, ids.class_weight_power, ids.max_class_weight)
        model = fit_mlp(s.train.X, s.train.y, s.val.X, s.val.y, ids.mlp, cw, params.seed)
    else:
        raise ValueError(f"unknown model {name!r}")
    return model, time.perf_counter() - t0


def train(
    params: Params,
    sample: Sample = "full",
    models: tuple[str, ...] = ALL_MODELS,
    register: bool = True,
    out_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    if "lightgbm" not in models:
        raise ValueError("lightgbm is required: it is the production model and the explainer")
    ids = params.ids
    out_dir = out_dir or params.resolve(Path("models/ids"))
    reports_dir = reports_dir or params.resolve(params.data.reports_dir) / "ids"
    figures = reports_dir / "figures"

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    s = load_splits(params, sample)
    log.info(
        "loaded %s splits: train %d, val %d, test %d",
        sample,
        len(s.train.y),
        len(s.val.y),
        len(s.test.y),
    )

    results: dict[str, dict[str, Any]] = {}
    scored: dict[str, Scored] = {}
    with mlflow.start_run(run_name=f"ids-train-{sample}") as parent:
        mlflow.log_params(
            {
                "sample": sample,
                "n_train": len(s.train.y),
                "n_features": len(s.feature_names),
                "class_weight_power": ids.class_weight_power,
                "benign_fpr_target": ids.benign_fpr_target,
                **{f"lgbm_{k}": v for k, v in ids.lightgbm.model_dump().items()},
                **{f"mlp_{k}": v for k, v in ids.mlp.model_dump().items()},
            }
        )
        for name in models:
            with mlflow.start_run(run_name=name, nested=True):
                model, secs = _fit(name, s, params)
                sc = score_model(model, s, ids.benign_fpr_target, secs)
                scored[name] = sc
                results[name] = {
                    "test": sc.test_metrics,
                    "threshold": sc.threshold,
                    "temperature": sc.calibration.temperature,
                    "train_seconds": secs,
                    "latency_ms_per_1k": sc.latency_ms_per_1k,
                }
                mlflow.log_metrics(
                    {
                        **flat(sc.test_metrics, "test_"),
                        "train_seconds": secs,
                        "latency_ms_per_1k": sc.latency_ms_per_1k,
                        "threshold": sc.threshold,
                    }
                )
                log.info(
                    "%s: test macro-F1 %.4f, benign FPR %.4f",
                    name,
                    sc.test_metrics["macro_f1"],
                    sc.test_metrics["benign_fpr"],
                )

        lg = scored["lightgbm"]
        assert isinstance(lg.model, LightGBMModel)
        mlp = scored.get("mlp")
        if mlp is not None:
            w, threshold = choose_ensemble_weight(
                lg.val_proba, mlp.val_proba, s.val.y, ids.benign_fpr_target
            )
            test_proba = w * lg.test_proba + (1 - w) * mlp.test_proba
        else:
            w, threshold, test_proba = 1.0, lg.threshold, lg.test_proba
        mlp_model = mlp.model if mlp is not None and w < 1 else None
        assert mlp_model is None or isinstance(mlp_model, MLPModel)
        bundle = IDSBundle(
            spec=s.spec,
            lgbm=lg.model,
            lgbm_calibration=lg.calibration,
            mlp=mlp_model,
            mlp_calibration=mlp.calibration if mlp_model is not None and mlp is not None else None,
            lgbm_weight=w,
            threshold=threshold,
            severity_bands=ids.severity_bands,
            metadata={"sample": sample, "parent_run_id": parent.info.run_id},
        )
        ens_metrics = evaluate(s.test.y, decide(test_proba, threshold), test_proba)
        ens_latency = _bundle_latency(bundle, s.test.X)
        results["ensemble"] = {
            "test": ens_metrics,
            "threshold": threshold,
            "lgbm_weight": w,
            "train_seconds": sum(
                r["train_seconds"] for k, r in results.items() if k in ("lightgbm", "mlp")
            ),
            "latency_ms_per_1k": ens_latency,
        }
        mlflow.log_metrics(
            {
                **flat(ens_metrics, "test_"),
                "lgbm_weight": w,
                "threshold": threshold,
                "latency_ms_per_1k": ens_latency,
            }
        )

        bundle.save(out_dir)
        _write_reports(results, s, lg.model, reports_dir, figures)
        mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
        version = log_bundle(out_dir, register=register)
        decision = promote(version, 1.5 * ids.benign_fpr_target) if version else None

    return {"models": results, "registry": decision}


def _bundle_latency(bundle: IDSBundle, X: np.ndarray) -> float:
    batch, best = X[:1000], np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        decide(bundle.proba(batch), bundle.threshold)
        best = min(best, time.perf_counter() - t0)
    return float(best * 1e6 / len(batch))


def _scalars(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = ("macro_f1", "benign_fpr", "attack_recall", "pr_auc_macro", "ece")
    return {
        name: {**{k: r["test"][k] for k in keys}, "latency_ms_per_1k": r["latency_ms_per_1k"]}
        for name, r in results.items()
    }


def _write_reports(
    results: dict[str, dict[str, Any]],
    s: Splits,
    lgbm: LightGBMModel,
    reports_dir: Path,
    figures: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    ens = results["ensemble"]["test"]
    plot_confusion(
        ens["confusion_matrix"], figures / "confusion_ensemble.png", "Ensemble — test split"
    )

    rng = np.random.default_rng(0)
    idx = rng.choice(len(s.test.y), size=min(SHAP_SAMPLE, len(s.test.y)), replace=False)
    # Keep every rare-class flow in the SHAP sample so each family gets a summary.
    rare = np.flatnonzero(
        np.isin(s.test.y, np.flatnonzero(np.bincount(s.test.y, minlength=len(CLASSES)) < 2000))
    )
    idx = np.unique(np.concatenate([idx, rare]))
    importance = global_importance(lgbm.contributions(s.test.X[idx]), s.test.y[idx])
    plot_global_importance(importance, s.feature_names, figures)
    top_features = {
        fam: [s.feature_names[i] for i in np.argsort(-imp)[:3]] for fam, imp in importance.items()
    }
    split_sizes = {"train": len(s.train.y), "val": len(s.val.y), "test": len(s.test.y)}
    (reports_dir / "model_card.md").write_text(
        model_card(results, top_features, split_sizes), encoding="utf-8"
    )

    md = [
        "# IDS results (test split)\n",
        comparison_table(results),
        "\n## Ensemble per-family\n",
        per_class_table(ens),
    ]
    (reports_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    (reports_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    (reports_dir / "metrics.json").write_text(
        json.dumps(_scalars(results), indent=2), encoding="utf-8"
    )
