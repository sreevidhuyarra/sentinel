"""FastAPI gateway: health, models, flow / email / URL scoring, alerts and the SOC copilot."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import Engine, func, insert, select

from sentinel.anomaly.bundle import AnomalyBundle
from sentinel.common.config import PROJECT_ROOT, get_settings
from sentinel.common.logging import get_logger
from sentinel.copilot.graph import Copilot
from sentinel.copilot.models import IncidentReport
from sentinel.copilot.tools import AlertTools
from sentinel.db.schema import alerts as alerts_table
from sentinel.db.schema import reports as reports_table
from sentinel.detection.fusion import Detector, load_review_delta
from sentinel.ids.bundle import IDSBundle
from sentinel.phishing.explain import explain
from sentinel.phishing.export import PhishingOnnxModel
from sentinel.phishing.text import extract_urls, model_input
from sentinel.phishing.urls import UrlModel
from sentinel.services.dashboard import router as dashboard_router
from sentinel.services.metrics import instrument, record_report

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
    needs_review: bool = False  # passed as benign, but the ensemble's members disagree
    member_gap: float | None = None


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
    log.warning("no model at %s; the endpoints that need it are disabled", path)
    return None, None


class Models(NamedTuple):
    detector: Detector
    ids_source: str
    anomaly_source: str | None
    phishing: PhishingOnnxModel | None
    phishing_source: str | None
    url: UrlModel | None
    url_source: str | None


def load_production_models() -> Models:
    from sentinel.anomaly.registry import load_bundle as load_anomaly
    from sentinel.ids.registry import load_bundle as load_ids
    from sentinel.phishing.registry import load_model as load_phishing
    from sentinel.phishing.registry import load_url_model

    s = get_settings()
    ids, ids_src = _load(s.ids_model_uri, s.ids_model_path, load_ids, IDSBundle.load, True)
    anomaly, an_src = _load(
        s.anomaly_model_uri, s.anomaly_model_path, load_anomaly, AnomalyBundle.load, False
    )
    phishing, ph_src = _load(
        s.phishing_model_uri, s.phishing_model_path, load_phishing, PhishingOnnxModel.load, False
    )
    url, url_src = _load(s.url_model_uri, s.url_model_path, load_url_model, UrlModel.load, False)
    assert ids is not None and ids_src is not None
    review = load_review_delta(PROJECT_ROOT / "reports" / "adversarial" / "results.json")
    detector = Detector(ids, anomaly, review_delta=review)
    return Models(detector, ids_src, an_src, phishing, ph_src, url, url_src)


MAX_EMAIL_URLS = 20
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
WS_POLL_SECONDS = 1.0
WS_MAX_BATCH = 200
WS_COLUMNS = [
    "id",
    "ts",
    "source",
    "src_ip",
    "dst_ip",
    "dst_port",
    "sender_domain",
    "predicted_family",
    "detector",
    "confidence",
    "severity",
]


def _db_engine(app: Any) -> Engine:
    """The API's read/write engine (created on first use unless injected)."""
    if app.state.db is None:
        from sentinel.db.schema import make_engine

        app.state.db = make_engine(get_settings().postgres_url)
    engine: Engine = app.state.db
    return engine


class UrlReason(BaseModel):
    feature: str
    contribution: float  # TreeSHAP log-odds pushed toward malicious
    text: str


class UrlScore(BaseModel):
    url: str
    is_malicious: bool
    probability: float
    reasons: list[UrlReason]


class UrlBatch(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=1000)
    explain: bool = True


class UrlResponse(BaseModel):
    threshold: float
    results: list[UrlScore]
    model_version: str | None


def score_urls(model: UrlModel, urls: list[str], explain_hits: bool) -> list[UrlScore]:
    if not urls:
        return []
    prob = model.predict_proba(urls)
    return [
        UrlScore(
            url=u,
            is_malicious=bool(p >= model.threshold),
            probability=float(p),
            reasons=[UrlReason(**r) for r in model.explain(u)]
            if explain_hits and p >= model.threshold
            else [],
        )
        for u, p in zip(urls, prob, strict=True)
    ]


class EmailIn(BaseModel):
    subject: str = ""
    body: str = Field(min_length=1, max_length=200_000)
    urls: list[str] = Field(default_factory=list, max_length=500)
    explain: bool = True


class EmailReason(BaseModel):
    text: str
    contribution: float  # drop in the malicious logit when this sentence is removed


class EmailScore(BaseModel):
    is_malicious: bool  # text model flagged OR any scored URL flagged
    flagged_by: list[str]  # subset of ["text", "url"]
    probability: float  # text model's P(phishing, fraud or spam)
    threshold: float
    severity: str | None
    reasons: list[EmailReason]  # sentences that raised the text score
    urls: list[str]  # URLs found in the email plus those supplied
    url_results: list[UrlScore]  # first 20 URLs, when the URL model is loaded
    model_version: str | None
    url_model_version: str | None


def create_app(
    bundle: IDSBundle | None = None,
    source: str = "injected",
    anomaly: AnomalyBundle | None = None,
    anomaly_source: str | None = None,
    phishing: PhishingOnnxModel | None = None,
    phishing_source: str | None = None,
    url: UrlModel | None = None,
    url_source: str | None = None,
    copilot: Copilot | None = None,
    db: Engine | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if bundle is None:
            m = load_production_models()
            app.state.detector, app.state.source, app.state.anomaly_source = (
                m.detector,
                m.ids_source,
                m.anomaly_source,
            )
            app.state.phishing, app.state.phishing_source = m.phishing, m.phishing_source
            app.state.url, app.state.url_source = m.url, m.url_source
        else:
            app.state.detector = Detector(bundle, anomaly)
            app.state.source = source
            app.state.anomaly_source = anomaly_source if anomaly else None
            app.state.phishing = phishing
            app.state.phishing_source = phishing_source if phishing else None
            app.state.url = url
            app.state.url_source = url_source if url else None
        app.state.copilot = copilot  # built on first investigation if None
        app.state.db = db
        yield

    app = FastAPI(title="Sentinel API", version="0.5.0", lifespan=lifespan)
    instrument(app)
    app.include_router(dashboard_router)

    @app.websocket("/ws/alerts")
    async def ws_alerts(ws: WebSocket) -> None:
        """Live alert feed: new rows of the alerts table, pushed as they are written.

        Polls the database once a second for ids above the last one sent, so any writer
        (stream detector, batch loader) shows up and the API needs no Kafka connection.
        Query `since_id` to resume; by default the feed starts at the newest alert.
        """
        await ws.accept()
        engine = _db_engine(ws.app)
        since = ws.query_params.get("since_id")
        with engine.connect() as conn:
            last = (
                int(since)
                if since
                else int(conn.execute(select(func.max(alerts_table.c.id))).scalar() or 0)
            )
        try:
            while True:
                with engine.connect() as conn:
                    rows = conn.execute(
                        select(*[alerts_table.c[c] for c in WS_COLUMNS])
                        .where(alerts_table.c.id > last)
                        .order_by(alerts_table.c.id)
                        .limit(WS_MAX_BATCH)
                    ).all()
                if rows:
                    last = int(rows[-1][0])
                    await ws.send_json(
                        {
                            "alerts": [
                                jsonable_encoder(dict(zip(WS_COLUMNS, r, strict=True)))
                                for r in rows
                            ]
                        }
                    )
                await asyncio.sleep(WS_POLL_SECONDS)
        except WebSocketDisconnect:
            return

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/app/" if FRONTEND_DIST.exists() else "/docs")

    @app.get("/app/{path:path}", include_in_schema=False)
    def frontend(path: str = "") -> FileResponse:
        """The built dashboard (frontend/dist); client routes fall back to index.html."""
        if not FRONTEND_DIST.exists():
            raise HTTPException(404, detail="dashboard not built: cd frontend && npm run build")
        target = (FRONTEND_DIST / path).resolve()
        if path and target.is_file() and FRONTEND_DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        d: Detector | None = getattr(request.app.state, "detector", None)
        return {
            "status": "ok",
            "model_loaded": d is not None,
            "anomaly_model_loaded": d is not None and d.anomaly is not None,
            "phishing_model_loaded": getattr(request.app.state, "phishing", None) is not None,
            "url_model_loaded": getattr(request.app.state, "url", None) is not None,
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
        ph: PhishingOnnxModel | None = request.app.state.phishing
        out["phishing"] = (
            {
                "source": request.app.state.phishing_source,
                "threshold": ph.threshold,
                "max_len": ph.max_len,
                "metadata": ph.metadata,
            }
            if ph is not None
            else None
        )
        um: UrlModel | None = request.app.state.url
        out["url"] = (
            {
                "source": request.app.state.url_source,
                "threshold": um.threshold,
                "metadata": um.metadata,
            }
            if um is not None
            else None
        )
        return out

    @app.post("/score/url", response_model=UrlResponse)
    def score_url(batch: UrlBatch, request: Request) -> UrlResponse:
        um: UrlModel | None = request.app.state.url
        if um is None:
            raise HTTPException(503, detail="URL model not loaded")
        return UrlResponse(
            threshold=um.threshold,
            results=score_urls(um, batch.urls, batch.explain),
            model_version=request.app.state.url_source,
        )

    @app.post("/score/phishing", response_model=EmailScore)
    def score_phishing(email: EmailIn, request: Request) -> EmailScore:
        ph: PhishingOnnxModel | None = request.app.state.phishing
        if ph is None:
            raise HTTPException(503, detail="phishing model not loaded")
        text = model_input(email.subject, email.body)
        prob = float(ph.predict_proba([text])[0])
        text_flag = prob >= ph.threshold
        urls = list(dict.fromkeys([*extract_urls(f"{email.subject}\n{email.body}"), *email.urls]))
        um: UrlModel | None = request.app.state.url
        url_results = score_urls(um, urls[:MAX_EMAIL_URLS], email.explain) if um else []
        # A link flags the email only above the validation-tuned email threshold; the
        # per-URL verdict (is_malicious) uses the stricter-on-false-alarms URL threshold.
        url_flag = (
            um is not None
            and um.email_threshold is not None
            and any(r.probability >= um.email_threshold for r in url_results)
        )
        flagged_by = [name for name, hit in (("text", text_flag), ("url", url_flag)) if hit]
        strong = prob >= 0.999 or any(r.probability >= 0.99 for r in url_results)
        reasons = explain(ph, text) if email.explain and text_flag else []
        return EmailScore(
            is_malicious=bool(flagged_by),
            flagged_by=flagged_by,
            probability=prob,
            threshold=ph.threshold,
            severity=("High" if strong or len(flagged_by) == 2 else "Medium")
            if flagged_by
            else None,
            reasons=[EmailReason(**r) for r in reasons],
            urls=urls,
            url_results=url_results,
            model_version=request.app.state.phishing_source,
            url_model_version=request.app.state.url_source,
        )

    def _db(request: Request) -> Engine:
        return _db_engine(request.app)

    def _copilot(request: Request) -> Copilot:
        if request.app.state.copilot is None:
            from sentinel.common.config import load_params
            from sentinel.copilot.service import build_copilot

            request.app.state.copilot = build_copilot(load_params())
        c: Copilot = request.app.state.copilot
        return c

    @app.get("/alerts")
    def list_alerts(
        request: Request,
        family: str | None = None,
        severity: str | None = None,
        source: str | None = None,
        src_ip: str | None = None,
        since: datetime | None = None,
        limit: int = Query(50, ge=1, le=100),
    ) -> dict[str, Any]:
        tools = AlertTools(_db(request))
        rows = tools.query_alerts(
            since=since,
            limit=limit,
            predicted_family=family,
            severity=severity,
            source=source,
            src_ip=src_ip,
        )
        return {"alerts": rows, "count": len(rows)}

    @app.get("/alerts/{alert_id}")
    def get_alert(alert_id: int, request: Request) -> dict[str, Any]:
        alert = AlertTools(_db(request)).get_alert(alert_id)
        if alert is None:
            raise HTTPException(404, detail="alert not found")
        return alert

    @app.post("/copilot/investigate/{alert_id}", response_model=IncidentReport)
    def investigate(alert_id: int, request: Request) -> IncidentReport:
        """Run the investigation graph (seconds with Gemini, minutes on the local model)."""
        from sentinel.copilot.graph import AlertNotFoundError

        t0 = time.perf_counter()
        try:
            report, trace = _copilot(request).investigate(alert_id)
        except AlertNotFoundError:
            raise HTTPException(404, detail="alert not found") from None
        record_report(report, time.perf_counter() - t0)
        with _db(request).begin() as conn:
            conn.execute(
                insert(reports_table).values(
                    id=report.report_id,
                    alert_id=alert_id,
                    created=datetime.now(UTC).replace(tzinfo=None),
                    status="done",
                    provider=report.provider,
                    report=report.model_dump(mode="json"),
                    trace=trace,
                )
            )
        return report

    @app.get("/reports/{report_id}", response_model=IncidentReport)
    def get_report(report_id: str, request: Request) -> IncidentReport:
        with _db(request).connect() as conn:
            row = conn.execute(
                select(reports_table.c.report).where(reports_table.c.id == report_id)
            ).first()
        if row is None:
            raise HTTPException(404, detail="report not found")
        return IncidentReport.model_validate(row[0])

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
