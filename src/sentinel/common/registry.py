"""MLflow model registry helpers shared by every model family.

Each family logs its bundle directory as a pyfunc model, gets a new version aliased
@staging, and moves @production only if it beats the current production version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.pyfunc.model import PythonModel
from mlflow.tracking import MlflowClient

from sentinel.common.logging import get_logger

log = get_logger(__name__)

SRC = Path(__file__).resolve().parents[1]  # src/sentinel
PIP_REQUIREMENTS = [
    "lightgbm",
    "torch",
    "polars",
    "numpy",
    "pandas",
    "scikit-learn",
    "pydantic",
]


def log_bundle(
    python_model: PythonModel, bundle_dir: Path, model_name: str, register: bool = True
) -> str | None:
    """Log the bundle to the active run; returns the registered version (or None)."""
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=python_model,
        artifacts={"bundle": str(bundle_dir)},
        code_paths=[str(SRC)],
        registered_model_name=model_name if register else None,
        pip_requirements=PIP_REQUIREMENTS,
    )
    version = getattr(info, "registered_model_version", None)
    return str(version) if version is not None else None


def promote(
    model_name: str,
    version: str,
    metric: str,
    fpr_metric: str,
    fpr_budget: float,
    production_value: float | None = None,
    gain_ci: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Alias `version` @staging; move @production to it only if it was trained and scored
    on the full data, keeps `fpr_metric` within `fpr_budget` and beats production on
    `metric` (higher is better).

    Scores are only comparable on the same test set, so dev-sample runs are never
    promoted, and a production model not trained on full data is always replaced.
    `production_value` is production re-scored on the candidate's test set, when that set
    differs from the one production was logged on (retraining drops overlapping rows).
    `gain_ci` is a bootstrap interval of the candidate's gain on those rows; when given, its
    lower end must be above zero, so a win within noise is not promoted.
    """
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "staging", version)
    run_id = client.get_model_version(model_name, version).run_id
    assert run_id is not None
    run = client.get_run(run_id)
    new_value = run.data.metrics.get(metric, 0.0)
    new_fpr = run.data.metrics.get(fpr_metric, 1.0)
    new_full = run.data.params.get("sample") == "full"
    try:
        prod = client.get_model_version_by_alias(model_name, "production")
        assert prod.run_id is not None
        prod_run = client.get_run(prod.run_id)
        prod_full = prod_run.data.params.get("sample") == "full"
        prod_value = prod_run.data.metrics.get(metric, 0.0) if prod_full else -1.0
        if production_value is not None and prod_full:
            prod_value = production_value
    except mlflow.exceptions.MlflowException:
        prod, prod_value = None, -1.0
    significant = gain_ci is None or prod is None or prod_value < 0 or gain_ci[0] > 0
    promoted = new_full and new_fpr <= fpr_budget and new_value > prod_value and significant
    if promoted:
        client.set_registered_model_alias(model_name, "production", version)
    decision = {
        "model": model_name,
        "version": version,
        "metric": metric,
        "candidate_sample": run.data.params.get("sample"),
        "candidate_value": new_value,
        "candidate_fpr": new_fpr,
        "production_version": prod.version if prod else None,
        "production_value": prod_value if prod and prod_value >= 0 else None,
        **({"gain_ci_low": gain_ci[0], "gain_ci_high": gain_ci[1]} if gain_ci else {}),
        "promoted": promoted,
    }
    log.info("promotion decision: %s", decision)
    return decision


def load_python_model(uri: str) -> Any:
    return mlflow.pyfunc.load_model(uri).unwrap_python_model()
