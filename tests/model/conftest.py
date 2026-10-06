"""Session fixtures: small IDS and anomaly models trained on the synthetic dataset, with a
throwaway SQLite MLflow registry shared by both."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from sentinel.common.config import (
    AnomalyParams,
    AutoencoderParams,
    IForestParams,
    MLPParams,
    Params,
    get_settings,
)


@contextmanager
def mlflow_sandbox(root: Path) -> Iterator[None]:
    """Point MLflow at `root` for the duration of a training call. MLflow's default
    artifact root is ./mlruns, so the working directory moves there too."""
    old_cwd, old_uri = Path.cwd(), os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    get_settings.cache_clear()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if old_uri is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = old_uri
        get_settings.cache_clear()


@pytest.fixture(scope="session")
def sandbox() -> Any:
    """`mlflow_sandbox` for tests that train again into the session registry."""
    return mlflow_sandbox


@pytest.fixture(scope="session")
def mlflow_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def small_params(built: Params) -> Params:
    ids = built.ids.model_copy(
        update={
            "lightgbm": built.ids.lightgbm.model_copy(update={"n_estimators": 60}),
            "xgboost": built.ids.xgboost.model_copy(update={"n_estimators": 60}),
            "mlp": MLPParams(hidden=[32, 16], epochs=3, patience=2, batch_size=256),
        }
    )
    anomaly = AnomalyParams(
        autoencoder=AutoencoderParams(
            hidden=[32, 16], bottleneck_sweep=[4, 8], epochs=8, patience=3, batch_size=256
        ),
        iforest=IForestParams(n_estimators=30, train_rows=2000),
    )
    return built.model_copy(update={"ids": ids, "anomaly": anomaly})


@pytest.fixture(scope="session")
def trained(small_params: Params, mlflow_root: Path) -> dict[str, Any]:
    from sentinel.ids.train import train

    root = mlflow_root
    with mlflow_sandbox(root):
        out = train(
            small_params,
            "full",
            register=True,
            out_dir=root / "bundle",
            reports_dir=root / "reports",
        )
    return {"out": out, "root": root, "params": small_params}


@pytest.fixture(scope="session")
def trained_anomaly(trained: dict[str, Any], mlflow_root: Path) -> dict[str, Any]:
    from sentinel.anomaly.train import train

    root = mlflow_root
    with mlflow_sandbox(root):
        out = train(
            trained["params"],
            register=True,
            out_dir=root / "anomaly",
            reports_dir=root / "anomaly_reports",
            ids_bundle_dir=root / "bundle",
            studies=("fusion", "holdout"),
        )
    return {"out": out, "root": root, "params": trained["params"]}
