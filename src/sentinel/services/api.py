"""FastAPI gateway: health, model info and batch flow scoring (supervised + anomaly)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from sentinel.anomaly.bundle import AnomalyBundle
from sentinel.common.config import get_settings
from sentinel.common.logging import get_logger
from sentinel.detection.fusion import Detector
from sentinel.ids.bundle import IDSBundle

log = get_logger(__name__)

MAX_BATCH = 10_000


class FlowBatch(BaseModel):
    flows: list[dict[str, float | int | None]] = Field(min_length=1, max_length=MAX_BATCH)
    explain: bool = True


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float
    text: str
    expected: float | None = None  # anomaly reasons: value of a normal-looking flow


class FlowScore(BaseModel):
    family: str
    is_attack: bool
    detector: str | None  # "supervised", "anomaly" or None (benign)
    confidence: float | None  # calibrated P(family); None for "Unknown anomaly"
    attack_score: float
    anomaly_score: float | None
    is_anomalous: bool | None
    severity: str | None
    probabilities: dict[str, float]
    reasons: list[Reason]


class ScoreResponse(BaseModel):
    model_version: str
    anomaly_model_version: str | None
    results: list[FlowScore]


def _load[T](
    uri: str,
    path: Path,
    from_registry: Callable[[str], T],
    from_dir: Callable[[Path], T],
    required: bool,
) -> tuple[T | None, str | None]:
    """Registry alias first; fall back to the last local training output."""
    try:
        import mlflow

        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        return from_registry(uri), uri
    except Exception as exc:  # registry unreachable or alias missing
        log.warning("registry load of %s failed (%s); trying %s", uri, exc, path)
    if path.exists() or required:
        return from_dir(path), str(path)
    log.warning("no anomaly model found; serving supervised verdicts only")
    return None, None


def load_production_models() -> tuple[Detector, str, str | None]:
    from sentinel.anomaly.registry import load_bundle as load_anomaly
    from sentinel.ids.registry import load_bundle as load_ids

    s = get_settings()
    ids, ids_src = _load(s.ids_model_uri, s.ids_model_path, load_ids, IDSBundle.load, True)
    anomaly, an_src = _load(
        s.anomaly_model_uri, s.anomaly_model_path, load_anomaly, AnomalyBundle.load, False
    )
    assert ids is not None and ids_src is not None
    return Detector(ids, anomaly), ids_src, an_src


def create_app(
    bundle: IDSBundle | None = None,
    source: str = "injected",
    anomaly: AnomalyBundle | None = None,
    anomaly_source: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if bundle is None:
            app.state.detector, app.state.source, app.state.anomaly_source = (
                load_production_models()
            )
        else:
            app.state.detector = Detector(bundle, anomaly)
            app.state.source = source
            app.state.anomaly_source = anomaly_source if anomaly else None
        yield

    app = FastAPI(title="Sentinel API", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        d: Detector | None = getattr(request.app.state, "detector", None)
        return {
            "status": "ok",
            "model_loaded": d is not None,
            "anomaly_model_loaded": d is not None and d.anomaly is not None,
        }

    @app.get("/models")
    def models(request: Request) -> dict[str, Any]:
        d: Detector = request.app.state.detector
        out: dict[str, Any] = {
            "ids": {
                "source": request.app.state.source,
                "threshold": d.ids.threshold,
                "lgbm_weight": d.ids.lgbm_weight,
                "severity_bands": d.ids.severity_bands,
                "n_features": len(d.ids.spec.columns),
                "metadata": d.ids.metadata,
            },
            "anomaly": None,
        }
        if d.anomaly is not None:
            out["anomaly"] = {
                "source": request.app.state.anomaly_source,
                "threshold": d.anomaly.threshold,
                "fpr_target": d.anomaly.fpr_target,
                "metadata": d.anomaly.metadata,
            }
        return out

    @app.post("/score/flows", response_model=ScoreResponse)
    def score_flows(batch: FlowBatch, request: Request) -> ScoreResponse:
        d: Detector = request.app.state.detector
        required = sorted(
            set(d.ids.spec.input_columns) | set(d.anomaly.spec.input_columns if d.anomaly else [])
        )
        missing = sorted({c for row in batch.flows for c in required if c not in row})
        if missing:
            raise HTTPException(422, detail={"missing_features": missing})
        df = pl.DataFrame(
            [{c: row[c] for c in required} for row in batch.flows],
            schema={c: pl.Float64 for c in required},
        )
        results = d.predict(df, explain=batch.explain)
        return ScoreResponse(
            model_version=str(request.app.state.source),
            anomaly_model_version=request.app.state.anomaly_source,
            results=[FlowScore.model_validate(r) for r in results],
        )

    return app
