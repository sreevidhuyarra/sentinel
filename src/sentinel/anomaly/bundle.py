"""Deployable anomaly detector: feature spec + autoencoder + threshold + explanations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sentinel.anomaly.autoencoder import AutoencoderModel
from sentinel.data.features import FeatureSpec
from sentinel.ids.explain import display_name

SEVERITY_MEDIUM_RATIO = 2.0


def _fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-6:
        return f"{round(v):,}"
    return f"{v:,.3g}" if abs(v) < 1e4 else f"{v:,.0f}"


def _unlog(values: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """Invert FeatureSpec's signed log1p on the log-compressed columns."""
    out = values.copy()
    for j, c in enumerate(spec.columns):
        if c in spec.log_columns:
            out[:, j] = np.sign(out[:, j]) * np.expm1(np.abs(out[:, j]))
    return out


@dataclass
class AnomalyBundle:
    spec: FeatureSpec
    autoencoder: AutoencoderModel
    threshold: float
    fpr_target: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def score(self, df: pl.DataFrame) -> np.ndarray:
        return self.autoencoder.score(self.spec.to_numpy(df))

    def severity(self, score: float) -> str:
        """Anomaly-only alerts carry no class probability, so they stay Low / Medium."""
        return "Medium" if score >= SEVERITY_MEDIUM_RATIO * self.threshold else "Low"

    def reasons(self, df: pl.DataFrame, top_k: int = 5) -> list[list[dict[str, Any]]]:
        """Features the autoencoder reconstructs worst, with the observed value and the
        value its closest normal-looking flow would have."""
        X = self.spec.to_numpy(df)
        err = self.autoencoder.feature_errors(X)
        observed = self.spec.raw(df)
        expected = _unlog(self.autoencoder.expected(X), self.spec)
        out = []
        for i in range(len(X)):
            top = np.argsort(-err[i])[:top_k]
            out.append(
                [
                    {
                        "feature": self.spec.columns[j],
                        "value": float(observed[i, j]),
                        "expected": float(expected[i, j]),
                        "contribution": float(err[i, j]),
                        "text": (
                            f"{display_name(self.spec.columns[j])} = {_fmt(observed[i, j])} is "
                            f"unusual for normal traffic (normal-looking value "
                            f"~ {_fmt(expected[i, j])})"
                        ),
                    }
                    for j in top
                ]
            )
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.spec.save(directory / "feature_spec.json")
        self.autoencoder.save(directory / "autoencoder")
        cfg = {
            "threshold": self.threshold,
            "fpr_target": self.fpr_target,
            "metadata": self.metadata,
        }
        (directory / "anomaly.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> AnomalyBundle:
        cfg = json.loads((directory / "anomaly.json").read_text(encoding="utf-8"))
        return cls(
            spec=FeatureSpec.load(directory / "feature_spec.json"),
            autoencoder=AutoencoderModel.load(directory / "autoencoder"),
            threshold=cfg["threshold"],
            fpr_target=cfg["fpr_target"],
            metadata=cfg.get("metadata", {}),
        )
