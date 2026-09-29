"""Isolation Forest baseline: the classical unsupervised detector the autoencoder must beat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from sentinel.common.config import IForestParams


@dataclass
class IForestModel:
    forest: IsolationForest

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (negated sklearn score_samples)."""
        return np.asarray(-self.forest.score_samples(X), dtype=np.float64)


def fit_iforest(X_benign: np.ndarray, p: IForestParams, seed: int = 42) -> IForestModel:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_benign), size=min(p.train_rows, len(X_benign)), replace=False)
    forest = IsolationForest(
        n_estimators=p.n_estimators, max_samples=p.max_samples, random_state=seed, n_jobs=-1
    )
    forest.fit(X_benign[idx])
    return IForestModel(forest)
