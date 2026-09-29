"""Registry wiring for the anomaly detector bundle (`anomaly-detector`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from mlflow.pyfunc.model import PythonModel

from sentinel.anomaly.bundle import AnomalyBundle
from sentinel.common import registry

MODEL_NAME = "anomaly-detector"
PROMOTION_METRIC = "test_roc_auc"


class AnomalyPyfunc(PythonModel):
    def load_context(self, context: Any) -> None:
        self.bundle = AnomalyBundle.load(Path(context.artifacts["bundle"]))

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        score = self.bundle.score(pl.from_pandas(model_input))
        return pd.DataFrame(
            {"anomaly_score": score, "is_anomalous": score >= self.bundle.threshold}
        )


def log_bundle(bundle_dir: Path, register: bool = True) -> str | None:
    return registry.log_bundle(AnomalyPyfunc(), bundle_dir, MODEL_NAME, register)


def promote(version: str, fpr_budget: float) -> dict[str, Any]:
    return registry.promote(MODEL_NAME, version, PROMOTION_METRIC, "test_benign_fpr", fpr_budget)


def load_bundle(uri: str = f"models:/{MODEL_NAME}@production") -> AnomalyBundle:
    wrapper: AnomalyPyfunc = registry.load_python_model(uri)
    return wrapper.bundle
