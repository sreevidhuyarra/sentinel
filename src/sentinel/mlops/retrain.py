"""Drift-triggered retraining as a Prefect flow (report section 9: Retraining).

pull labelled live flows -> retrain (training split + live) -> evaluate on the fixed test
split -> register @staging -> promote to @production only if it beats production within the
false-alarm budget (sentinel.common.registry.promote). The candidate bundle is written to
its own directory; the production bundle on disk is only replaced when promoted.
"""

from __future__ import annotations

import json
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE
from sqlalchemy import Engine, insert, select, update

from sentinel.common.config import Params, get_settings, load_params
from sentinel.db.schema import flows, init_db, make_engine, retrain_runs
from sentinel.ids.dataset import CLASSES

CANDIDATE_DIR = Path("models/ids_candidate")


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# Inputs (a DB engine, a DataFrame) are not hashable, and results must never be reused.
@task(name="pull-labelled-live-flows", cache_policy=NONE)
def labelled_live_flows(engine: Engine, inputs: list[str], limit: int) -> pl.DataFrame:
    """The most recent sampled flows that carry a label, as model inputs + `family`."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(flows.c.features, flows.c.family)
            .where(flows.c.family.is_not(None))
            .order_by(flows.c.id.desc())
            .limit(limit)
        ).all()
    known = set(CLASSES)
    data = []
    for feats, family in rows:
        if family not in known:
            continue
        f = feats if isinstance(feats, dict) else json.loads(feats)
        data.append({**{c: f.get(c) for c in inputs}, "family": family})
    schema: dict[str, Any] = {c: pl.Float64 for c in inputs} | {"family": pl.String}
    return pl.DataFrame(data, schema=schema)


@task(name="retrain-and-evaluate", cache_policy=NONE)
def retrain_ids(
    params: Params, extra: pl.DataFrame, run_name: str, baseline_dir: Path
) -> dict[str, Any]:
    from sentinel.ids.bundle import IDSBundle
    from sentinel.ids.train import train

    return train(
        params,
        sample="full",
        models=tuple(params.mlops.retrain_models),
        register=True,
        out_dir=params.resolve(CANDIDATE_DIR),
        reports_dir=params.resolve(Path("reports/ids_retrain")),
        extra_train=extra,
        run_name=run_name,
        # The local bundle is kept in step with @production (copied on every promotion).
        baseline=IDSBundle.load(baseline_dir),
    )


@flow(name="sentinel-retrain", log_prints=True)
def retrain_flow(reason: str, params_file: str | None = None) -> dict[str, Any]:
    params = load_params(Path(params_file) if params_file else None)
    logger = get_run_logger()
    settings = get_settings()
    engine = make_engine(settings.postgres_url)
    init_db(engine)
    with engine.begin() as conn:
        key = conn.execute(
            insert(retrain_runs).values(started=_now(), reason=reason, status="running")
        ).inserted_primary_key
        assert key is not None
        run_id = int(key[0])
    try:
        from sentinel.ids.bundle import IDSBundle

        inputs = IDSBundle.load(params.resolve(Path("models/ids"))).spec.input_columns
        extra = labelled_live_flows(engine, sorted(inputs), params.mlops.max_live_rows)
        logger.info(
            "labelled live flows: %d (%s)",
            len(extra),
            extra["family"].value_counts().to_dicts() if len(extra) else "none",
        )
        if len(extra) < params.mlops.min_live_rows:
            out: dict[str, Any] = {"registry": None, "models": {}}
            status = "skipped"
            logger.info(
                "too few labelled live flows (< %d); not retraining", params.mlops.min_live_rows
            )
        else:
            out = retrain_ids(
                params, extra, f"ids-retrain-{run_id}", params.resolve(Path("models/ids"))
            )
            decision = out.get("registry") or {}
            status = "promoted" if decision.get("promoted") else "staged"
            if decision.get("promoted"):
                # The detector reloads @production from MLflow; keep the local fallback in step.
                shutil.copytree(
                    params.resolve(CANDIDATE_DIR),
                    params.resolve(Path("models/ids")),
                    dirs_exist_ok=True,
                )
        decision = out.get("registry") or {}
        with engine.begin() as conn:
            conn.execute(
                update(retrain_runs)
                .where(retrain_runs.c.id == run_id)
                .values(
                    finished=_now(),
                    status=status,
                    n_extra=len(extra),
                    version=decision.get("version"),
                    metrics={k: v for k, v in decision.items() if k != "model"} or None,
                )
            )
        logger.info("retrain %s: %s", status, decision)
        return {"id": run_id, "status": status, "n_extra": len(extra), "decision": decision}
    except Exception as exc:
        with engine.begin() as conn:
            conn.execute(
                update(retrain_runs)
                .where(retrain_runs.c.id == run_id)
                .values(finished=_now(), status="failed", error=traceback.format_exc()[-4000:])
            )
        logger.error("retrain failed: %s", exc)
        raise
