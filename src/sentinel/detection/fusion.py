"""Fuse the supervised classifier with the anomaly detector (report section 5.2).

Rule: the supervised verdict stands when it raises an alert. When it says Benign but the
autoencoder cannot reconstruct the flow (score above threshold), the flow becomes an
"Unknown anomaly" alert: something unlike normal traffic that matches no known family.
Both scores are always returned so the analyst and the copilot can see them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from sentinel.anomaly.bundle import AnomalyBundle
from sentinel.ids.bundle import IDSBundle

UNKNOWN = "Unknown anomaly"


@dataclass
class Detector:
    ids: IDSBundle
    anomaly: AnomalyBundle | None = None

    def predict(
        self, df: pl.DataFrame, explain: bool = True, top_k: int = 5
    ) -> list[dict[str, Any]]:
        results = self.ids.predict(df, explain=explain, top_k=top_k)
        for r in results:
            r["detector"] = "supervised" if r["is_attack"] else None
            r["anomaly_score"] = None
            r["is_anomalous"] = None
        if self.anomaly is None or not results:
            return results

        score = self.anomaly.score(df)
        anomalous = score >= self.anomaly.threshold
        promote = [i for i, r in enumerate(results) if anomalous[i] and not r["is_attack"]]
        reasons = self.anomaly.reasons(df[promote], top_k) if explain and promote else []
        for i, r in enumerate(results):
            r["anomaly_score"] = float(score[i])
            r["is_anomalous"] = bool(anomalous[i])
        for j, i in enumerate(promote):
            r = results[i]
            r.update(
                family=UNKNOWN,
                is_attack=True,
                # The supervised confidence here is P(its own class), i.e. of Benign.
                confidence=None,
                detector="anomaly",
                severity=self.anomaly.severity(float(score[i])),
                reasons=reasons[j] if reasons else [],
            )
        return results

    @property
    def anomaly_threshold(self) -> float | None:
        return self.anomaly.threshold if self.anomaly else None


def fused_flags(sup_flag: np.ndarray, anomaly_score: np.ndarray, threshold: float) -> np.ndarray:
    """Vectorised version of the rule for evaluation: alert if either detector fires."""
    return np.asarray(sup_flag | (anomaly_score >= threshold))
