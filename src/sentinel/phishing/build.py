"""Stage: Kaggle LoRA output -> deployable ONNX bundle, before/after benchmark, MLflow.

Accuracy on the full validation and test splits comes from the Kaggle run's predictions
(PyTorch, float32 weights). The exported models are then compared with the PyTorch model
on a fixed sample of test emails (the laptop CPU scores ~8 emails/s, so the full 36k test
split per model would take over an hour): quality (F1, ROC-AUC, verdict agreement), size,
single-email latency (p50/p95) and batched throughput.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from sentinel.common.config import Params, get_settings
from sentinel.common.logging import get_logger
from sentinel.phishing.export import PhishingOnnxModel, export_onnx, merge_adapter, quantize_int8

log = get_logger(__name__)

EXPERIMENT = "sentinel-phishing"
LATENCY_EMAILS = 200


def threshold_at_budget(val: pd.DataFrame, budget: float) -> float:
    """Score exceeded by at most `budget` of legitimate validation emails. method="higher"
    lands on an actual score, so interpolation cannot push the rate over budget."""
    q = np.quantile(val.loc[val["label"] == 0, "prob"], 1 - budget, method="higher")
    return float(np.nextafter(q, np.inf))


def split_metrics(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    y, p = df["label"].to_numpy(), df["prob"].to_numpy()
    flag = p >= threshold
    pr, rc, f1, _ = precision_recall_fscore_support(y, flag, average="binary", zero_division=0)
    return {
        "f1": float(f1),
        "precision": float(pr),
        "recall": float(rc),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "false_alarm_rate": float(flag[y == 0].mean()),
        "flagged_by_kind": {
            k: float(flag[df["kind"] == k].mean()) for k in sorted(df["kind"].unique())
        },
        "by_source": {
            s: _source_metrics(flag, y, (df["source"] == s).to_numpy())
            for s in sorted(df["source"].unique())
        },
    }


def _source_metrics(flag: np.ndarray, y: np.ndarray, m: np.ndarray) -> dict[str, Any]:
    legit, bad = m & (y == 0), m & (y == 1)
    return {
        "emails": int(m.sum()),
        "false_alarm_rate": float(flag[legit].mean()) if legit.any() else None,
        "recall": float(flag[bad].mean()) if bad.any() else None,
    }


def _latency(score: Any, texts: list[str]) -> dict[str, float]:
    times = []
    for t in texts:
        t0 = time.perf_counter()
        score([t])
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(times)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}


def _throughput(score: Any, texts: list[str]) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    p = score(texts)
    return p, len(texts) / (time.perf_counter() - t0)


def benchmark(
    torch_model: Any,
    tokenizer: Any,
    onnx_files: dict[str, Path],
    bundle: Path,
    sample: pd.DataFrame,
    max_len: int,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    import torch

    texts, y = sample["text"].tolist(), sample["label"].to_numpy()

    @torch.no_grad()
    def torch_score(batch: list[str], batch_size: int = 32) -> np.ndarray:
        out = np.empty(len(batch))
        order = np.argsort([len(t) for t in batch])
        for i in range(0, len(batch), batch_size):
            idx = order[i : i + batch_size]
            enc = tokenizer(
                [batch[j] for j in idx],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            )
            out[idx] = torch.softmax(torch_model(**enc).logits, 1)[:, 1].numpy()
        return out

    results: dict[str, dict[str, Any]] = {}
    ref, tput = _throughput(torch_score, texts)
    results["pytorch_fp32"] = {
        "size_mb": None,
        "emails_per_s": tput,
        **_latency(torch_score, texts[:LATENCY_EMAILS]),
    }
    for name, path in onnx_files.items():
        m = PhishingOnnxModel.load(bundle, model_file=str(path.resolve()))
        p, tput = _throughput(m.predict_proba, texts)
        results[name] = {
            "size_mb": path.stat().st_size / 1e6,
            "emails_per_s": tput,
            **_latency(m.predict_proba, texts[:LATENCY_EMAILS]),
            "verdicts_differ_from_pytorch": int(((p >= threshold) != (ref >= threshold)).sum()),
            "max_prob_diff": float(np.abs(p - ref).max()),
            "_probs": p,
        }
    for r, p in [(results["pytorch_fp32"], ref)] + [
        (results[n], results[n].pop("_probs")) for n in onnx_files
    ]:
        r.update(f1=float(f1_score(y, p >= threshold)), roc_auc=float(roc_auc_score(y, p)))
    return results


def build(params: Params, register: bool = True) -> dict[str, Any]:
    from transformers import AutoTokenizer

    p = params.phishing
    kaggle = params.resolve(p.kaggle_output_dir)
    adapter = kaggle / "adapter"
    bundle = params.resolve(Path("models/phishing/bundle"))
    work = params.resolve(Path("models/phishing/work"))
    reports = params.resolve(params.data.reports_dir) / "phishing"
    for d in (bundle, work, reports):
        d.mkdir(parents=True, exist_ok=True)

    run_info = json.loads((kaggle / "run_info.json").read_text(encoding="utf-8"))
    if run_info.get("smoke_test"):
        raise ValueError("Kaggle output is from a smoke test; rerun the notebook on Kaggle")
    preds = {s: pd.read_csv(kaggle / f"{s}_predictions.csv") for s in ("val", "test")}
    threshold = threshold_at_budget(preds["val"], p.fpr_budget)
    accuracy = {s: split_metrics(df, threshold) for s, df in preds.items()}
    at_half = {s: split_metrics(df, 0.5) for s, df in preds.items()}

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model = merge_adapter(adapter)
    fp32 = export_onnx(model, tokenizer, work / "model_fp32.onnx", p.max_len)
    files = {"onnx_fp32": fp32}
    for mode in ("embeddings", "full"):
        files[f"onnx_int8_{mode}"] = quantize_int8(fp32, work / f"model_int8_{mode}.onnx", mode)
    deployed_key = "onnx_fp32" if p.quantization == "none" else f"onnx_int8_{p.quantization}"
    shutil.copy2(files[deployed_key], bundle / "model.onnx")
    tokenizer.save_pretrained(bundle / "tokenizer")
    cfg = {
        "model_file": "model.onnx",
        "max_len": p.max_len,
        "threshold": threshold,
        "fpr_budget": p.fpr_budget,
        "quantization": p.quantization,
        "metadata": {"base_model": run_info["config"]["model_name"], "lora": run_info["config"]},
    }
    (bundle / "phishing.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    test = pd.read_parquet(params.resolve(p.kaggle_dir) / "test.parquet")
    sample = test.sample(min(p.benchmark_emails, len(test)), random_state=params.seed)
    bench = benchmark(model, tokenizer, files, bundle, sample, p.max_len, threshold)

    results = {
        "threshold": threshold,
        "fpr_budget": p.fpr_budget,
        "deployed": deployed_key,
        "accuracy": accuracy,
        "accuracy_at_0.5": at_half,
        "benchmark_sample": len(sample),
        "benchmark": bench,
        "training": {
            "minutes": run_info["train_minutes"],
            "gpus": run_info["environment"]["cuda_devices"],
        },
    }
    (reports / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    from sentinel.phishing.report import model_card, results_markdown

    (reports / "results.md").write_text(results_markdown(results), encoding="utf-8")
    (reports / "model_card.md").write_text(model_card(results), encoding="utf-8")
    metrics = {
        k: accuracy["test"][k] for k in ("f1", "precision", "recall", "pr_auc", "false_alarm_rate")
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    decision = None
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="phishing-build"):
        mlflow.log_params(
            {
                "sample": "full",
                "threshold": threshold,
                "fpr_budget": p.fpr_budget,
                "quantization": p.quantization,
                "max_len": p.max_len,
                **{f"lora_{k}": v for k, v in run_info["config"].items()},
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_metrics(
            {f"val_{k}": accuracy["val"][k] for k in ("f1", "pr_auc", "false_alarm_rate")}
        )
        for name, b in bench.items():
            mlflow.log_metrics(
                {f"bench_{name}_{k}": float(v) for k, v in b.items() if isinstance(v, int | float)}
            )
        mlflow.log_artifacts(str(reports), artifact_path="reports")
        from sentinel.phishing.registry import log_bundle, promote

        version = log_bundle(bundle, register=register)
        if version:
            decision = promote(version, 2 * p.fpr_budget)
    results["registry"] = decision
    log.info("phishing bundle built: %s", decision)
    return results
