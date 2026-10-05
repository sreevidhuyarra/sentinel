"""`sentinel` command-line entry point. Each subcommand is one reproducible pipeline stage."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from sentinel.common.config import load_params
from sentinel.common.logging import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Data pipeline: ingest, build, synthetic sample.")
app.add_typer(data_app, name="data")


@app.callback()
def main(log_level: str = "INFO", json_logs: bool = False) -> None:
    configure_logging(log_level, json_logs)


@data_app.command()
def download(params_file: Path | None = None) -> None:
    """Download the corrected CIC-IDS2017 CSVs into data/raw/cicids2017."""
    from sentinel.data.download import download_cicids2017

    p = load_params(params_file)
    for path in download_cicids2017(p.resolve(p.data.raw_dir)):
        typer.echo(path)


@data_app.command()
def ingest(raw_dir: Path | None = None, params_file: Path | None = None) -> None:
    """Raw CSVs -> data/interim/flows.parquet."""
    from sentinel.data.ingest import ingest as run

    p = load_params(params_file)
    run(raw_dir or p.resolve(p.data.raw_dir), p.resolve(p.data.interim_dir) / "flows.parquet")


@data_app.command()
def build(params_file: Path | None = None) -> None:
    """Clean, label, split and validate -> data/processed/, feature spec, reference sample."""
    from sentinel.data.pipeline import build as run

    summary = run(load_params(params_file))
    typer.echo(json.dumps(summary["families"], indent=2))


@data_app.command()
def synth(out_dir: Path = Path("data/raw/synthetic"), seed: int = 0) -> None:
    """Write a small synthetic dataset in the real CSV format (used by tests and CI)."""
    from sentinel.data.synthetic import generate

    for path in generate(out_dir, seed):
        typer.echo(path)


UNSW_CSV = Path("data/raw/cic_unsw_nb15/CICFlowMeter_out.csv")
ids_app = typer.Typer(no_args_is_help=True, help="Module 1: supervised intrusion detection.")
app.add_typer(ids_app, name="ids")


@ids_app.command("train")
def ids_train(
    sample: str = typer.Option("full", help="full | dev"),
    models: str = typer.Option("logreg,lightgbm,xgboost,mlp", help="comma-separated"),
    register: bool = typer.Option(True, help="register the bundle in the MLflow model registry"),
    params_file: Path | None = None,
) -> None:
    """Train, calibrate, evaluate and register the IDS models; writes reports/ids/."""
    from sentinel.ids.train import train

    if sample not in ("full", "dev"):
        raise typer.BadParameter("sample must be 'full' or 'dev'")
    out = train(load_params(params_file), sample, tuple(models.split(",")), register)  # type: ignore[arg-type]
    for name, r in out["models"].items():
        m = r["test"]
        typer.echo(f"{name:10s} macro-F1 {m['macro_f1']:.4f}  benign FPR {m['benign_fpr']:.4%}")
    typer.echo(json.dumps(out["registry"], indent=2))


@ids_app.command("tune")
def ids_tune(
    family: str = typer.Option("lightgbm", help="lightgbm | xgboost"),
    trials: int | None = None,
    sample: str = typer.Option("full", help="full (real class mix) | dev (quick check)"),
    params_file: Path | None = None,
) -> None:
    """Optuna search scored on validation; prints the best parameters."""
    from sentinel.ids.tune import tune

    if family not in ("lightgbm", "xgboost"):
        raise typer.BadParameter("family must be 'lightgbm' or 'xgboost'")
    if sample not in ("full", "dev"):
        raise typer.BadParameter("sample must be 'full' or 'dev'")
    out = tune(load_params(params_file), family, trials, sample)  # type: ignore[arg-type]
    typer.echo(json.dumps(out, indent=2))


@ids_app.command("cross-dataset")
def ids_cross_dataset(
    csv: Path | None = typer.Option(None, help=f"CIC-UNSW-NB15 flow CSV [default: {UNSW_CSV}]"),
    download: bool = typer.Option(False, help="download the CSV first if missing"),
    params_file: Path | None = None,
) -> None:
    """Train on CIC-IDS2017 (shared features), score UNSW-NB15 unchanged; writes a report."""
    from sentinel.ids.crossdataset import run

    p = load_params(params_file)
    csv = p.resolve(csv or UNSW_CSV)
    if download:
        from sentinel.data.download import download_cic_unsw_nb15

        csv = download_cic_unsw_nb15(csv.parent)
    run(p, csv)
    typer.echo("wrote reports/ids/cross_dataset.md")


@ids_app.command("imbalance")
def ids_imbalance(params_file: Path | None = None) -> None:
    """Compare class weighting, resampling and focal loss on the dev sample."""
    from sentinel.ids.imbalance import study

    study(load_params(params_file))
    typer.echo("wrote reports/ids/imbalance.md")


anomaly_app = typer.Typer(no_args_is_help=True, help="Module 2: anomaly detection.")
app.add_typer(anomaly_app, name="anomaly")


@anomaly_app.command("train")
def anomaly_train(
    register: bool = typer.Option(True, help="register the bundle in the MLflow model registry"),
    studies: str = typer.Option("fusion,holdout,unsw", help="comma-separated; '' for none"),
    params_file: Path | None = None,
) -> None:
    """Train autoencoder + Isolation Forest on benign flows; run the Module 2 studies."""
    from sentinel.anomaly.train import train

    chosen = tuple(x for x in studies.split(",") if x)
    out = train(load_params(params_file), register=register, studies=chosen)
    for name in ("autoencoder", "iforest"):
        t = out[name]["test"]
        typer.echo(
            f"{name:12s} ROC-AUC {t['roc_auc']:.4f}  benign FPR {t['benign_fpr']:.2%}  "
            f"attack recall {t['attack_recall']:.2%}"
        )
    typer.echo(json.dumps(out["registry"], indent=2))


phishing_app = typer.Typer(no_args_is_help=True, help="Module 3: phishing email classifier.")
app.add_typer(phishing_app, name="phishing")


@phishing_app.command("download")
def phishing_download(params_file: Path | None = None) -> None:
    """Download the 11 email corpora (Zenodo 8339691, ~430 MB) into data/raw/phishing_emails."""
    from sentinel.phishing.data import download

    p = load_params(params_file)
    for path in download(p.resolve(p.phishing.raw_dir)):
        typer.echo(path)


@phishing_app.command("prepare")
def phishing_prepare(params_file: Path | None = None) -> None:
    """Clean, deduplicate and split the emails; write the Kaggle upload folder."""
    from sentinel.phishing.data import build

    p = load_params(params_file)
    summary = build(p)
    typer.echo(json.dumps(summary["by_kind"], indent=2))
    typer.echo(f"Kaggle upload folder: {p.resolve(p.phishing.kaggle_dir)}")


@phishing_app.command("build")
def phishing_build(
    register: bool = typer.Option(True, help="register the bundle in the MLflow model registry"),
    params_file: Path | None = None,
) -> None:
    """Kaggle LoRA output -> merged ONNX int8 bundle, before/after benchmark, MLflow."""
    from sentinel.phishing.build import build

    out = build(load_params(params_file), register=register)
    t = out["accuracy"]["test"]
    typer.echo(
        f"test F1 {t['f1']:.4f}  false alarms {t['false_alarm_rate']:.2%}  "
        f"(threshold {out['threshold']:.3f})"
    )
    typer.echo("wrote reports/phishing/results.md")
    typer.echo(json.dumps(out["registry"], indent=2))


@phishing_app.command("url-train")
def phishing_url_train(
    register: bool = typer.Option(True, help="register the bundle in the MLflow model registry"),
    params_file: Path | None = None,
) -> None:
    """Train the URL classifier; evaluate it alone, cross-dataset and combined with email text."""
    from sentinel.phishing.url_train import train

    out = train(load_params(params_file), register=register)
    t = out["test"]
    typer.echo(
        f"URL test ROC-AUC {t['roc_auc']:.4f}  false alarms {t['false_alarm_rate']:.2%}  "
        f"caught {t['recall']:.2%}"
    )
    typer.echo("wrote reports/phishing/url_results.md")
    typer.echo(json.dumps(out["registry"], indent=2))


@phishing_app.command("notebook")
def phishing_notebook(
    src: Path = typer.Option(Path("notebooks/phishing_lora_kaggle.py")),
    dst: Path = typer.Option(Path("notebooks/phishing_lora_kaggle.ipynb")),
) -> None:
    """Build the Kaggle fine-tuning notebook (.ipynb) from its tested .py source."""
    from sentinel.phishing.notebook import script_to_notebook

    typer.echo(script_to_notebook(src, dst))


adversarial_app = typer.Typer(no_args_is_help=True, help="Module 4: adversarial robustness.")
app.add_typer(adversarial_app, name="adversarial")


@adversarial_app.command("run")
def adversarial_run(params_file: Path | None = None) -> None:
    """Attack the deployed detectors, train the defenses, write reports/adversarial/."""
    from sentinel.adversarial.run import run

    out = run(load_params(params_file))
    typer.echo(json.dumps(out["problem_space"], indent=2))
    typer.echo("wrote reports/adversarial/results.md")


alerts_app = typer.Typer(no_args_is_help=True, help="Alert database (Postgres).")
app.add_typer(alerts_app, name="alerts")


@alerts_app.command("load")
def alerts_load(
    source: str = typer.Option("all", help="network, email or all"),
    split: str = typer.Option("test", help="test (the alert DB) or val (the dev set)"),
    database: str | None = typer.Option(
        None, help="Postgres database (default: from .env for test, copilot.dev_database for val)"
    ),
    params_file: Path | None = None,
) -> None:
    """Score a split with the deployed detectors and (re)fill that database's alerts table.

    The validation split goes to a separate database, so tuning the copilot on it never
    changes the related activity seen by test alerts (the splits interleave in time).
    """
    import polars as pl

    from sentinel.common.config import get_settings
    from sentinel.db import load
    from sentinel.db.schema import ensure_database, init_db, make_engine, with_database
    from sentinel.services.api import load_production_models

    if split not in ("test", "val"):
        raise typer.BadParameter("split must be test or val")
    p, s = load_params(params_file), get_settings()
    db = database or (p.copilot.dev_database if split == "val" else None)
    url = with_database(s.postgres_url, db)
    ensure_database(url)
    engine = make_engine(url)
    init_db(engine, s.postgres_ro_password)
    m = load_production_models()
    if source in ("network", "all"):
        load.clear(engine, "network")
        flows = pl.read_parquet(p.resolve(p.data.processed_dir) / "flows.parquet").filter(
            pl.col("split") == split
        )
        typer.echo(f"network alerts: {load.load_network(engine, m.detector, flows, m.ids_source)}")
    if source in ("email", "all"):
        if m.phishing is None:
            raise typer.BadParameter("phishing model not available")
        load.clear(engine, "email")
        emails = pl.read_parquet(p.resolve(p.phishing.processed_dir) / "emails.parquet").filter(
            pl.col("split") == split
        )
        n = load.load_email(
            engine,
            m.phishing,
            m.url,
            emails,
            p.copilot.email_sample if split == "test" else p.copilot.dev_email_sample,
            p.seed,
            m.phishing_source or "local",
        )
        typer.echo(f"email alerts: {n}")


copilot_app = typer.Typer(no_args_is_help=True, help="Module 5: SOC copilot.")
app.add_typer(copilot_app, name="copilot")


@copilot_app.command("kb")
def copilot_kb(download: bool = False, params_file: Path | None = None) -> None:
    """Build the ATT&CK + KEV knowledge base and the hybrid retrieval index."""
    from sentinel.copilot import kb
    from sentinel.copilot.rag import Retriever
    from sentinel.copilot.service import RAG_DIR

    p = load_params(params_file)
    raw, out = p.resolve(Path("data/raw/knowledge")), p.resolve(Path("data/processed/knowledge"))
    if download or not (raw / "enterprise-attack.json").exists():
        kb.download(raw)
    typer.echo(json.dumps(kb.build(raw, out)))
    Retriever.build(out, p.resolve(RAG_DIR))


@copilot_app.command("guard-train")
def copilot_guard_train(params_file: Path | None = None) -> None:
    """Train the injection guard's classifier; write reports/copilot/guard_results.json."""
    from sentinel.copilot.guard_train import run

    out = run(load_params(params_file))
    typer.echo(
        json.dumps({"chosen": out["chosen"], "test": out["test"]["rules+classifier"]["all"]})
    )


@copilot_app.command("guard-rethreshold")
def copilot_guard_rethreshold(params_file: Path | None = None) -> None:
    """Re-choose the trained guard's threshold on validation for the current scoring
    (e.g. after enabling sentence-pair scoring), then score the guard test set once."""
    from sentinel.copilot.guard_train import rethreshold

    out = rethreshold(load_params(params_file))
    typer.echo(
        json.dumps(
            {
                "previous": out["previous"]["test"]["rules+classifier"]["all"],
                "now": out["test"]["rules+classifier"]["all"],
            },
            indent=2,
        )
    )


def _providers(provider: str | None) -> list[str] | None:
    return [x.strip() for x in provider.split(",")] if provider else None


@copilot_app.command("investigate")
def copilot_investigate(
    alert_id: int,
    provider: str | None = typer.Option(None, help="e.g. ollama or gemini,ollama"),
    params_file: Path | None = None,
) -> None:
    """Investigate one alert and print the incident report as JSON."""
    from sentinel.copilot.service import build_copilot
    from sentinel.llm.factory import build_provider

    p = load_params(params_file)
    llm = build_provider(p, order=_providers(provider)) if provider else "config"
    report, trace = build_copilot(p, llm=llm).investigate(alert_id)
    typer.echo(report.model_dump_json(indent=2))
    typer.echo(json.dumps(trace), err=True)


@copilot_app.command("evaluate")
def copilot_evaluate(
    limit: int | None = typer.Option(None, help="first N gold alerts only"),
    reports: bool = typer.Option(True, help="also generate reports (uses the LLM)"),
    judge: bool = typer.Option(True, help="LLM-as-judge faithfulness"),
    provider: str | None = typer.Option(None, help="generator providers, e.g. ollama"),
    judge_provider: str | None = typer.Option(
        None, help="judge providers (default: gemini judge model)"
    ),
    fresh: bool = typer.Option(False, help="discard the resume checkpoint and start over"),
    dev: bool = typer.Option(False, help="development alert set (tuning) -> reports/copilot/dev"),
    baseline: bool = typer.Option(
        False, help="pipeline before the improvement pass, for comparison"
    ),
    params_file: Path | None = None,
) -> None:
    """Gold-set evaluation: retrieval, technique mapping, faithfulness, cost (reports/copilot).

    Resumable: finished reports are checkpointed, so rerunning after an interruption skips them.
    """
    from sentinel.copilot.run_eval import run

    out = run(
        load_params(params_file),
        limit,
        reports,
        judge,
        _providers(provider),
        _providers(judge_provider),
        fresh,
        dev,
        baseline,
    )
    typer.echo(json.dumps(out.get("reports", {}), indent=2, default=str))


@copilot_app.command("redteam")
def copilot_redteam(
    n: int = typer.Option(24, help="alerts to attack (half email, half network)"),
    provider: str | None = typer.Option(None, help="e.g. ollama"),
    params_file: Path | None = None,
) -> None:
    """Prompt-injection attack success with and without the guard (reports/copilot)."""
    from sentinel.copilot.run_eval import run_redteam

    out = run_redteam(load_params(params_file), n, _providers(provider))
    typer.echo(json.dumps(out["summary"], indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI gateway (loads ids-classifier@production from MLflow)."""
    import uvicorn

    from sentinel.services.api import create_app

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
