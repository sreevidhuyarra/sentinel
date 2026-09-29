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
