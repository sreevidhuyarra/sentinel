"""Run the copilot studies and write reports/copilot/ (gold set, results, red-team)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from sentinel.common.config import Params, get_settings
from sentinel.common.logging import get_logger
from sentinel.copilot import evaluate, redteam
from sentinel.copilot.gold import build_gold_set
from sentinel.copilot.report import write_markdown
from sentinel.copilot.service import build_copilot
from sentinel.db.schema import alert_truth, alerts, make_engine
from sentinel.llm.factory import build_judge, build_provider

log = get_logger(__name__)

REPORTS = Path("reports/copilot")


def _round_robin(gold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order so that any prefix covers the labels evenly (for --limit runs)."""
    by: dict[str, list[dict[str, Any]]] = {}
    for g in gold:
        by.setdefault(g["label"], []).append(g)
    out: list[dict[str, Any]] = []
    while any(by.values()):
        for label in list(by):
            if by[label]:
                out.append(by[label].pop(0))
    return out


def _log_mlflow(name: str, metrics: dict[str, Any], params: dict[str, Any]) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        mlflow.set_experiment("sentinel-copilot")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
            )
    except Exception as exc:  # MLflow down: results are on disk anyway
        log.warning("MLflow logging skipped: %s", exc)


def run(
    params: Params,
    limit: int | None = None,
    reports: bool = True,
    judge: bool = True,
    provider: list[str] | None = None,
    judge_provider: list[str] | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    out_dir = params.resolve(REPORTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    admin = make_engine(get_settings().postgres_url)  # reads labels; the copilot cannot
    gold = _round_robin(build_gold_set(admin, params.seed))
    (out_dir / "gold_set.json").write_text(evaluate.dump(gold))
    log.info("gold set: %d alerts, %d labels", len(gold), len({g["label"] for g in gold}))

    llm = build_provider(params, order=provider) if provider else "config"
    copilot = build_copilot(params, llm=llm if reports else None)
    results: dict[str, Any] = {"gold_alerts": len(gold)}
    results["retrieval"] = evaluate.evaluate_retrieval(copilot, gold)

    if reports:
        subset = gold[:limit] if limit else gold
        j = build_judge(params, order=judge_provider) if judge else None
        generator = getattr(copilot.llm, "name", "none").replace("+", "_")
        checkpoint = out_dir / f"eval_rows.{generator}.partial.jsonl"
        if fresh:
            checkpoint.unlink(missing_ok=True)
        rows = evaluate.evaluate_reports(
            copilot,
            subset,
            j,
            checkpoint=checkpoint,
            progress=lambda i, r: log.info(
                "report %d/%d alert %s %s p@1=%s",
                i + 1,
                len(subset),
                r["alert_id"],
                r["label"],
                r["p@1"],
            ),
        )
        results["reports"] = evaluate.summarise(rows)
        results["reports"]["generator"] = getattr(copilot.llm, "name", None)
        results["reports"]["judge"] = None if j is None else f"{j.name}:{j.model}"
        by_label: dict[str, list[bool]] = {}
        for r in rows:
            by_label.setdefault(r["label"], []).append(bool(r["p@1"]))
        results["reports"]["p@1_by_label"] = {k: float(np.mean(v)) for k, v in by_label.items()}
        (out_dir / "eval_rows.json").write_text(evaluate.dump(rows))
        _log_mlflow(
            "evaluate",
            {k: v for k, v in results["reports"].items() if isinstance(v, float)},
            {"n": len(rows), "generator": results["reports"]["generator"]},
        )
    (out_dir / "eval_results.json").write_text(evaluate.dump(results))
    write_markdown(out_dir)
    return results


def redteam_alerts(params: Params, n: int) -> list[int]:
    """Half email alerts (true phishing / fraud), half network alerts across families."""
    admin = make_engine(get_settings().postgres_url)
    rng = np.random.default_rng(params.seed + 1)
    with admin.connect() as conn:
        email = [
            r[0]
            for r in conn.execute(
                select(alerts.c.id)
                .join(alert_truth, alert_truth.c.alert_id == alerts.c.id)
                .where(alerts.c.source == "email", alert_truth.c.label.in_(["phishing", "fraud"]))
                .order_by(alerts.c.id)
            )
        ]
        network: list[int] = []
        fams = ["BruteForce", "WebAttack", "PortScan", "DoS", "DDoS", "Bot"]
        for fam in fams:
            ids = [
                r[0]
                for r in conn.execute(
                    select(alerts.c.id)
                    .where(alerts.c.predicted_family == fam)
                    .order_by(alerts.c.id)
                )
            ]
            k = max(1, (n // 2) // len(fams))
            network += [
                int(ids[i]) for i in sorted(rng.choice(len(ids), min(k, len(ids)), replace=False))
            ]
    email_ids = [
        int(email[i])
        for i in sorted(rng.choice(len(email), min(n - len(network), len(email)), replace=False))
    ]
    return email_ids + network


def run_redteam(params: Params, n: int, provider: list[str] | None = None) -> dict[str, Any]:
    out_dir = params.resolve(REPORTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = build_provider(params, order=provider) if provider else "config"
    copilot = build_copilot(params, llm=llm)
    out = redteam.run(copilot, redteam_alerts(params, n), params.seed)
    out["generator"] = getattr(copilot.llm, "name", None)
    (out_dir / "redteam.json").write_text(json.dumps(out, indent=2, default=str))
    _log_mlflow(
        "redteam",
        {f"asr_{k}": v["attack_success_rate"] for k, v in out["summary"].items()},
        {"cases": len(out["cases"]), "generator": out["generator"]},
    )
    write_markdown(out_dir)
    return out
