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


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI gateway (loads ids-classifier@production from MLflow)."""
    import uvicorn

    from sentinel.services.api import create_app

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
