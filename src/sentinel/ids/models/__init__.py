"""Model wrappers. Each exposes `logits(X) -> (n, n_classes)` raw scores; probabilities
come only from the calibration layer applied on top."""

from typing import Protocol

import numpy as np


class LogitModel(Protocol):
    def logits(self, X: np.ndarray) -> np.ndarray: ...
