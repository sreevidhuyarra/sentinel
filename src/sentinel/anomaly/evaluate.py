"""Anomaly-score evaluation: thresholds from benign validation, detection per family."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(benign_scores: np.ndarray, fpr: float) -> float:
    """Score above which `fpr` of benign flows fall (the report's 99th-percentile rule)."""
    return float(np.nextafter(np.quantile(benign_scores, 1 - fpr), np.inf))


def detection(
    score: np.ndarray, threshold: float, labels: np.ndarray, benign_label: str = "Benign"
) -> dict[str, Any]:
    """Benign-vs-attack quality of an anomaly score and recall for every attack label."""
    is_attack = labels != benign_label
    flagged = score >= threshold
    out: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(is_attack, score)) if 0 < is_attack.mean() < 1 else None,
        "pr_auc": float(average_precision_score(is_attack, score)) if is_attack.any() else None,
        "benign_fpr": float(flagged[~is_attack].mean()),
        "attack_recall": float(flagged[is_attack].mean()) if is_attack.any() else None,
        "per_label_recall": {},
    }
    for lab in sorted(set(labels) - {benign_label}):
        m = labels == lab
        out["per_label_recall"][lab] = {"recall": float(flagged[m].mean()), "flows": int(m.sum())}
    return out


def recall_at_fprs(
    benign_val: np.ndarray, score: np.ndarray, labels: np.ndarray, fprs: list[float]
) -> dict[str, dict[str, float]]:
    """Attack recall and realised benign FPR on the evaluation set for each target FPR,
    thresholds taken from benign validation scores."""
    is_attack = labels != "Benign"
    out = {}
    for f in fprs:
        t = threshold_at_fpr(benign_val, f)
        flagged = score >= t
        out[f"{f:g}"] = {
            "threshold": t,
            "benign_fpr": float(flagged[~is_attack].mean()),
            "attack_recall": float(flagged[is_attack].mean()),
        }
    return out
