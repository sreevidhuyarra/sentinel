"""Class weights for imbalanced training."""

from __future__ import annotations

import numpy as np

from sentinel.ids.dataset import CLASSES


def class_weights(y: np.ndarray, power: float = 0.5, cap: float = 1000.0) -> np.ndarray:
    """Weight per class = (n_largest / n_class) ** power, capped at `cap`.

    power=1 is inverse frequency: with ~860k benign and ~20 infiltration flows that
    gives one infiltration flow the weight of 40,000 benign ones, which overfits the
    handful of rare examples. power=0.5 (square root) is the usual compromise.
    """
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    w = np.ones(len(CLASSES))
    present = counts > 0
    w[present] = (counts.max() / counts[present]) ** power
    return np.asarray(np.minimum(w, cap))


def sample_weights(y: np.ndarray, power: float = 0.5, cap: float = 1000.0) -> np.ndarray:
    return np.asarray(class_weights(y, power, cap)[y], dtype=np.float32)
