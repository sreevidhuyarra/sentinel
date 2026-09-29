"""Class-weighted logistic regression: the sanity floor every other model must beat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LogRegModel:
    pipeline: Pipeline

    def logits(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.pipeline.decision_function(X), dtype=np.float64)


def fit_logreg(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None, seed: int = 42
) -> LogRegModel:
    pipe = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=300, C=1.0, random_state=seed)
    )
    pipe.fit(X, y, logisticregression__sample_weight=sample_weight)
    return LogRegModel(pipe)
