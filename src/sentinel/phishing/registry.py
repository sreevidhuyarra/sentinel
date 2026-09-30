"""Registry wiring for the phishing email classifier (`phishing-classifier`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.pyfunc.model import PythonModel

from sentinel.common import registry
from sentinel.phishing.export import PhishingOnnxModel

MODEL_NAME = "phishing-classifier"
PROMOTION_METRIC = "test_pr_auc"


class PhishingPyfunc(PythonModel):
    """Input: DataFrame with a `text` column (see sentinel.phishing.text.model_input)."""

    def load_context(self, context: Any) -> None:
        self.model = PhishingOnnxModel.load(Path(context.artifacts["bundle"]))

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        prob = self.model.predict_proba(model_input["text"].tolist())
        return pd.DataFrame({"prob": prob, "is_malicious": prob >= self.model.threshold})


def log_bundle(bundle_dir: Path, register: bool = True) -> str | None:
    return registry.log_bundle(PhishingPyfunc(), bundle_dir, MODEL_NAME, register)


def promote(version: str, fpr_budget: float) -> dict[str, Any]:
    return registry.promote(
        MODEL_NAME, version, PROMOTION_METRIC, "test_false_alarm_rate", fpr_budget
    )


def load_model(uri: str = f"models:/{MODEL_NAME}@production") -> PhishingOnnxModel:
    wrapper: PhishingPyfunc = registry.load_python_model(uri)
    return wrapper.model


URL_MODEL_NAME = "url-classifier"


class UrlPyfunc(PythonModel):
    """Input: DataFrame with a `url` column."""

    def load_context(self, context: Any) -> None:
        from sentinel.phishing.urls import UrlModel

        self.model = UrlModel.load(Path(context.artifacts["bundle"]))

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        prob = self.model.predict_proba(model_input["url"].astype(str).tolist())
        return pd.DataFrame({"prob": prob, "is_malicious": prob >= self.model.threshold})


def log_url_bundle(bundle_dir: Path, register: bool = True) -> str | None:
    return registry.log_bundle(UrlPyfunc(), bundle_dir, URL_MODEL_NAME, register)


def promote_url(version: str, fpr_budget: float) -> dict[str, Any]:
    # Judged on Hannousse, whose legitimate URLs look like real links. Overall test PR-AUC is
    # dominated by PhiUSIIL's bare-homepage legitimate URLs, which every model separates
    # almost perfectly, so it cannot tell a usable model from one with many false alarms.
    return registry.promote(
        URL_MODEL_NAME,
        version,
        "test_Hannousse_pr_auc",
        "test_Hannousse_false_alarm_rate",
        fpr_budget,
    )


def load_url_model(uri: str = f"models:/{URL_MODEL_NAME}@production") -> Any:
    wrapper: UrlPyfunc = registry.load_python_model(uri)
    return wrapper.model
