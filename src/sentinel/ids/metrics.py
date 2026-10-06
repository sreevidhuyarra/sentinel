"""Evaluation metrics for imbalanced multiclass flow classification.

Accuracy is reported but never used for selection: with ~78% benign traffic a model
that flags nothing scores 0.78.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from sentinel.ids.dataset import BENIGN, CLASSES


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 over all classes via one bincount. Called every boosting round during
    early stopping, where sklearn's f1_score would dominate training time."""
    k = len(CLASSES)
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def paired_bootstrap_f1(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap of macro-F1(a) - macro-F1(b) over the same rows.

    Resampling rows only changes how often each (true, pred_a, pred_b) cell occurs, so each
    resample is one multinomial draw over at most k^3 cells: exact, and milliseconds on the
    full test split. Returns the observed gap, its (1 - alpha) percentile interval and the
    share of resamples where `a` is not better.
    """
    k = len(CLASSES)
    code = (y * k + pred_a) * k + pred_b
    counts = np.bincount(code, minlength=k**3)
    cells = np.flatnonzero(counts)
    draws = (
        np.random.default_rng(seed)
        .multinomial(len(y), counts[cells] / len(y), size=n_boot)
        .astype(np.float64)
    )

    def f1(cm_index: np.ndarray) -> np.ndarray:
        flat_cm = np.zeros((n_boot, k * k))
        np.add.at(flat_cm.T, cm_index, draws.T)
        cm = flat_cm.reshape(n_boot, k, k)
        tp = np.diagonal(cm, axis1=1, axis2=2)
        denom = cm.sum(axis=1) + cm.sum(axis=2)
        out = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
        return np.asarray(out.mean(axis=1))

    true = cells // (k * k)
    diff = f1(true * k + (cells // k) % k) - f1(true * k + cells % k)
    low, high = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    return {
        "gain": macro_f1(y, pred_a) - macro_f1(y, pred_b),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_not_better": float((diff <= 0).mean()),
    }


def benign_fpr(y: np.ndarray, pred: np.ndarray) -> float:
    benign = y == BENIGN
    return float((pred[benign] != BENIGN).mean()) if benign.any() else 0.0


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, bins: int = 15) -> float:
    """Top-label ECE: gap between confidence and accuracy, averaged over confidence bins."""
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def evaluate(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    """Full metric set. Scalar metrics are flat keys (MLflow-friendly); per-class and the
    confusion matrix are nested for reports."""
    labels = list(range(len(CLASSES)))
    per_class_f1 = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    recall = np.divide(np.diag(cm), support, out=np.zeros(len(labels)), where=support > 0)
    precision = np.divide(np.diag(cm), predicted, out=np.zeros(len(labels)), where=predicted > 0)
    pr_auc = {}
    for i, c in enumerate(CLASSES):
        if support[i] > 0:
            pr_auc[c] = float(average_precision_score(y == i, proba[:, i]))
    return {
        "macro_f1": macro_f1(y, pred),
        "benign_fpr": benign_fpr(y, pred),
        "attack_recall": float((pred[y != BENIGN] != BENIGN).mean())
        if (y != BENIGN).any()
        else 0.0,
        "pr_auc_macro": float(np.mean(list(pr_auc.values()))),
        "ece": expected_calibration_error(y, proba),
        "accuracy": float((y == pred).mean()),
        "per_class": {
            c: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(per_class_f1[i]),
                "pr_auc": pr_auc.get(c),
                "support": int(support[i]),
            }
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }


def flat(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Scalar metrics plus per-class recall/F1, flattened for mlflow.log_metrics."""
    out = {f"{prefix}{k}": float(v) for k, v in metrics.items() if isinstance(v, int | float)}
    for c, m in metrics.get("per_class", {}).items():
        out[f"{prefix}recall_{c}"] = m["recall"]
        out[f"{prefix}f1_{c}"] = m["f1"]
    return out
