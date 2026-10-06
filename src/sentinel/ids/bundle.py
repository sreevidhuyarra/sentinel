"""The deployable IDS model: everything needed to go from raw flow rows to explained alerts.

A bundle is a directory (registered in MLflow as one artifact) holding the feature spec,
LightGBM booster, MLP weights, per-model calibration, ensemble weight, decision threshold
and severity bands. Training writes it; the API and the streaming detector load it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sentinel.data.features import FeatureSpec
from sentinel.ids.calibration import TemperatureBias
from sentinel.ids.dataset import BENIGN, CLASSES
from sentinel.ids.decision import attack_score, decide, severity
from sentinel.ids.explain import top_reasons
from sentinel.ids.models.gbm import LightGBMModel
from sentinel.ids.models.mlp import MLPModel


@dataclass
class IDSBundle:
    spec: FeatureSpec
    lgbm: LightGBMModel
    lgbm_calibration: TemperatureBias
    mlp: MLPModel | None
    mlp_calibration: TemperatureBias | None
    lgbm_weight: float
    threshold: float
    severity_bands: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba(X)[0]

    def _proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Ensemble probabilities, and the members' attack-probability gap (if two members)."""
        p = self.lgbm_calibration.apply(self.lgbm.logits(X))
        if self.mlp is not None and self.mlp_calibration is not None and self.lgbm_weight < 1:
            p_mlp = self.mlp_calibration.apply(self.mlp.logits(X))
            gap = np.abs((1 - p[:, BENIGN]) - (1 - p_mlp[:, BENIGN]))
            return self.lgbm_weight * p + (1 - self.lgbm_weight) * p_mlp, gap
        return p, None

    def predict(
        self, df: pl.DataFrame, explain: bool = True, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Score raw flow rows (canonical column names). Reasons are computed for alerts only."""
        if df.height == 0:
            return []
        X = self.spec.to_numpy(df)
        proba, gap = self._proba(X)
        pred = decide(proba, self.threshold)
        score = attack_score(proba)
        sev = severity(score, self.severity_bands)
        alerts = np.flatnonzero(pred != BENIGN)
        reasons: dict[int, list[dict[str, Any]]] = {}
        if explain and len(alerts):
            contrib = self.lgbm.contributions(X[alerts])
            raw = self.spec.raw(df[alerts.tolist()])
            for j, i in enumerate(alerts):
                reasons[int(i)] = top_reasons(
                    contrib[j], raw[j], self.spec.columns, int(pred[i]), top_k
                )
        return [
            {
                "family": CLASSES[pred[i]],
                "is_attack": bool(pred[i] != BENIGN),
                "confidence": float(proba[i, pred[i]]),
                "attack_score": float(score[i]),
                "severity": sev[i] if pred[i] != BENIGN else None,
                "probabilities": {c: float(proba[i, k]) for k, c in enumerate(CLASSES)},
                "reasons": reasons.get(i, []),
                # |P_lgbm(attack) - P_mlp(attack)|: the review flag's disagreement signal.
                "member_gap": None if gap is None else float(gap[i]),
            }
            for i in range(df.height)
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.spec.save(directory / "feature_spec.json")
        self.lgbm.save(directory / "lightgbm.txt")
        if self.mlp is not None:
            self.mlp.save(directory / "mlp")
        config = {
            "classes": CLASSES,
            "lgbm_calibration": self.lgbm_calibration.as_dict(),
            "mlp_calibration": self.mlp_calibration.as_dict() if self.mlp_calibration else None,
            "lgbm_weight": self.lgbm_weight,
            "threshold": self.threshold,
            "severity_bands": self.severity_bands,
            "metadata": self.metadata,
        }
        (directory / "bundle.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> IDSBundle:
        cfg = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
        if cfg["classes"] != CLASSES:
            raise ValueError(f"Bundle classes {cfg['classes']} do not match code {CLASSES}")
        has_mlp = (directory / "mlp").exists()
        return cls(
            spec=FeatureSpec.load(directory / "feature_spec.json"),
            lgbm=LightGBMModel.load(directory / "lightgbm.txt"),
            lgbm_calibration=TemperatureBias(**cfg["lgbm_calibration"]),
            mlp=MLPModel.load(directory / "mlp") if has_mlp else None,
            mlp_calibration=TemperatureBias(**cfg["mlp_calibration"])
            if cfg["mlp_calibration"]
            else None,
            lgbm_weight=cfg["lgbm_weight"],
            threshold=cfg["threshold"],
            severity_bands=cfg["severity_bands"],
            metadata=cfg.get("metadata", {}),
        )
