"""Turn calibrated probabilities into alerts under a benign false-positive budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel.ids.dataset import BENIGN
from sentinel.ids.metrics import benign_fpr, macro_f1

SEVERITIES = ["Low", "Medium", "High", "Critical"]


def attack_score(proba: np.ndarray) -> np.ndarray:
    """Probability that a flow is *any* attack."""
    return np.asarray(1.0 - proba[:, BENIGN])


def decide(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Benign unless the attack score clears `threshold`; then the most likely attack family.

    Separating "is it an attack?" from "which attack?" lets the threshold control benign
    FPR directly, which plain argmax cannot.
    """
    attacks = proba.copy()
    attacks[:, BENIGN] = -np.inf
    pred = attacks.argmax(axis=1)
    pred[attack_score(proba) < threshold] = BENIGN
    return np.asarray(pred)


@dataclass
class ThresholdChoice:
    threshold: float
    val_macro_f1: float
    val_benign_fpr: float


def choose_threshold(proba: np.ndarray, y: np.ndarray, fpr_target: float) -> ThresholdChoice:
    """Threshold with the best validation macro-F1 among those with benign FPR <= target."""
    benign_scores = attack_score(proba[y == BENIGN])
    candidates = np.unique(
        np.concatenate(
            [
                np.quantile(benign_scores, 1 - np.linspace(0, fpr_target, 41)),
                np.linspace(0.05, 0.99, 48),
            ]
        )
    )
    # Nudge each candidate just above a benign score so ties don't break the budget.
    candidates = np.nextafter(candidates, np.inf)
    best: ThresholdChoice | None = None
    for t in candidates:
        pred = decide(proba, float(t))
        fpr = benign_fpr(y, pred)
        if fpr > fpr_target:
            continue
        f1 = macro_f1(y, pred)
        if best is None or f1 > best.val_macro_f1:
            best = ThresholdChoice(float(t), f1, fpr)
    if best is None:  # every candidate over budget: be as conservative as possible
        t = float(candidates.max())
        pred = decide(proba, t)
        best = ThresholdChoice(t, macro_f1(y, pred), benign_fpr(y, pred))
    return best


def severity(score: np.ndarray, bands: list[float]) -> list[str]:
    """Map attack scores to Low / Medium / High / Critical using ascending cut points."""
    idx = np.searchsorted(np.asarray(bands), score, side="right")
    return [SEVERITIES[i] for i in idx]
