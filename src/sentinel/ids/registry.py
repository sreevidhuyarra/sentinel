"""Registry wiring for the supervised IDS bundle (`ids-classifier`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from mlflow.pyfunc.model import PythonModel

from sentinel.common import registry
from sentinel.ids.bundle import IDSBundle

MODEL_NAME = "ids-classifier"
PROMOTION_METRIC = "test_macro_f1"


class IDSPyfunc(PythonModel):
    """Lets any MLflow consumer (mlflow models serve, batch scoring) use the bundle."""

    def load_context(self, context: Any) -> None:
        self.bundle = IDSBundle.load(Path(context.artifacts["bundle"]))

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        rows = self.bundle.predict(pl.from_pandas(model_input), explain=False)
        keys = ("family", "is_attack", "confidence", "attack_score", "severity")
        return pd.DataFrame([{k: r[k] for k in keys} for r in rows])


def log_bundle(bundle_dir: Path, register: bool = True) -> str | None:
    return registry.log_bundle(IDSPyfunc(), bundle_dir, MODEL_NAME, register)


def promote(
    version: str,
    fpr_budget: float,
    production_value: float | None = None,
    gain_ci: tuple[float, float] | None = None,
) -> dict[str, Any]:
    return registry.promote(
        MODEL_NAME,
        version,
        PROMOTION_METRIC,
        "test_benign_fpr",
        fpr_budget,
        production_value,
        gain_ci,
    )


def load_bundle(uri: str = f"models:/{MODEL_NAME}@production") -> IDSBundle:
    wrapper: IDSPyfunc = registry.load_python_model(uri)
    return wrapper.bundle
