"""Train and evaluate the URL classifier, and the combined email + URL verdict.

Protocol: URLs are deduplicated on their normalised form and split by registered domain
(StratifiedGroupKFold over source x label: 3 folds train, 1 validation, 1 test), so no
website appears in two splits. Early stopping and the threshold use validation only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from sentinel.common.config import Params, get_settings
from sentinel.common.logging import get_logger
from sentinel.phishing.urls import UrlModel, normalize, registered_domain

log = get_logger(__name__)

EXPERIMENT = "sentinel-url"
MAX_URLS_PER_EMAIL = 20
LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "auc",
    "verbosity": -1,
}


def load_urls(raw_dir: Path) -> pd.DataFrame:
    phi = pd.read_csv(next(raw_dir.glob("PhiUSIIL*.csv")), usecols=["URL", "label"])
    phi = pd.DataFrame({"url": phi["URL"], "label": 1 - phi["label"], "source": "PhiUSIIL"})
    han = pd.concat(
        [pd.read_parquet(raw_dir / f"hannousse_{s}.parquet") for s in ("train", "test")]
    )
    han = pd.DataFrame(
        {
            "url": han["url"],
            "label": (han["status"] == "phishing").astype(int),
            "source": "Hannousse",
        }
    )
    df = pd.concat([phi, han], ignore_index=True).dropna(subset=["url"])
    df["url"] = df["url"].astype(str).str.strip()
    df["norm"] = df["url"].map(normalize)
    conflict = df.groupby("norm")["label"].nunique()
    df = (
        df[~df["norm"].isin(conflict[conflict > 1].index)]
        .drop_duplicates("norm")
        .reset_index(drop=True)
    )
    df["domain"] = df["url"].map(registered_domain)
    return df


def split(df: pd.DataFrame, seed: int, n_folds: int = 5) -> pd.DataFrame:
    folds = np.empty(len(df), dtype=np.int64)
    strat = df["source"] + "_" + df["label"].astype(str)
    for k, (_, idx) in enumerate(
        StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(
            df, strat, df["domain"]
        )
    ):
        folds[idx] = k
    df = df.copy()
    df["split"] = np.select(
        [folds < n_folds - 2, folds == n_folds - 2], ["train", "val"], default="test"
    )
    return df


def source_weights(df: pd.DataFrame) -> np.ndarray:
    """Weights giving every source the same total weight. PhiUSIIL is ~20x larger than
    Hannousse, and its legitimate URLs are all bare homepages, so unweighted training
    learns "has a path -> malicious" (99.5% false alarms on Hannousse legitimate URLs)."""
    counts = df["source"].value_counts()
    w = (len(df) / (len(counts) * counts)).to_dict()
    return np.asarray(df["source"].map(w).to_numpy(), dtype=np.float64)


def fit(
    train: pd.DataFrame, val: pd.DataFrame, max_features: int, seed: int, balance: bool = False
) -> UrlModel:
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vec.fit(train["url"].map(normalize))
    shell = UrlModel(vec, None, 0.5, {})
    Xtr, Xva = shell.features(train["url"].tolist()), shell.features(val["url"].tolist())
    w_tr = source_weights(train) if balance else None
    w_va = source_weights(val) if balance else None
    booster = lgb.train(
        {**LGB_PARAMS, "seed": seed},
        lgb.Dataset(Xtr, train["label"].to_numpy(), weight=w_tr),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(Xva, val["label"].to_numpy(), weight=w_va)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    shell.booster = booster
    return shell


def threshold_at_budget(benign_scores: np.ndarray, budget: float) -> float:
    return float(np.nextafter(np.quantile(benign_scores, 1 - budget, method="higher"), np.inf))


def metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    flag = p >= t
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, flag)),
        "false_alarm_rate": float(flag[y == 0].mean()),
        "recall": float(flag[y == 1].mean()),
    }


def _email_frame(params: Params, model: UrlModel, split_name: str) -> pd.DataFrame:
    """Email split with the text model's probability and the highest URL score per email
    (-1 when the email has no URL)."""
    emails = pl.read_parquet(
        params.resolve(params.phishing.processed_dir) / "emails.parquet"
    ).filter(pl.col("split") == split_name)
    kaggle = params.resolve(params.phishing.kaggle_output_dir)
    text = pd.read_csv(kaggle / f"{split_name}_predictions.csv")
    df = (
        emails.select("id", "label", "kind", "urls")
        .to_pandas()
        .merge(text[["id", "prob"]], on="id")
    )
    flat = [(i, u) for i, urls in enumerate(df["urls"]) for u in list(urls)[:MAX_URLS_PER_EMAIL]]
    best = np.full(len(df), -1.0)
    if flat:
        p = model.predict_proba([u for _, u in flat])
        for (i, _), pi in zip(flat, p, strict=True):
            best[i] = max(best[i], pi)
    df["max_url_prob"] = best
    df.attrs["urls_scored"] = len(flat)
    return df


def _verdict_metrics(df: pd.DataFrame, flag: np.ndarray) -> dict[str, Any]:
    y = df["label"].to_numpy()
    return {
        "false_alarm_rate": float(flag[y == 0].mean()),
        "recall": float(flag[y == 1].mean()),
        "f1": float(f1_score(y, flag)),
        "caught_by_kind": {
            k: float(flag[(df["kind"] == k).to_numpy()].mean())
            for k in sorted(df["kind"].unique())
            if k != "legitimate"
        },
    }


def _fusion_features(df: pd.DataFrame) -> np.ndarray:
    t = np.clip(df["prob"].to_numpy(), 1e-6, 1 - 1e-6)
    u = df["max_url_prob"].to_numpy()
    has = (u >= 0).astype(float)
    uc = np.clip(np.where(u < 0, 0.5, u), 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(t / (1 - t)), has * np.log(uc / (1 - uc)), has])


def learned_fusion(val: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Fairer test than "text OR link": logistic regression over [text logit, best-link logit,
    has-link] fitted on one half of the validation emails; for each false-alarm budget the
    threshold of both it and text alone is set on the other half, then both are scored on
    test. If links carried much extra signal, the fused score would catch clearly more."""
    idx = np.random.default_rng(seed).permutation(len(val))
    a, b = val.iloc[idx[: len(idx) // 2]], val.iloc[idx[len(idx) // 2 :]]
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(_fusion_features(a), a["label"])
    y = test["label"].to_numpy()
    rows = []
    for budget in (0.005, 0.01, 0.02):
        for name, s_b, s_t in (
            ("text only", b["prob"].to_numpy(), test["prob"].to_numpy()),
            (
                "learned text + link",
                lr.predict_proba(_fusion_features(b))[:, 1],
                lr.predict_proba(_fusion_features(test))[:, 1],
            ),
        ):
            t = threshold_at_budget(s_b[b["label"].to_numpy() == 0], budget)
            flag = s_t >= t
            rows.append(
                {
                    "budget": budget,
                    "method": name,
                    "false_alarm_rate": float(flag[y == 0].mean()),
                    "recall": float(flag[y == 1].mean()),
                    "phishing": float(flag[(test["kind"] == "phishing").to_numpy()].mean()),
                }
            )
    return {
        "weights": dict(
            zip(["text_logit", "link_logit", "has_link"], lr.coef_[0].tolist(), strict=True)
        ),
        "rows": rows,
    }


def email_study(params: Params, model: UrlModel) -> dict[str, Any]:
    """Text model alone vs text OR any link flagged.

    The URL threshold used inside the email verdict is chosen on the email VALIDATION split:
    the lowest one (most extra detections) that raises validation false alarms by at most
    `url_email_extra_fpr` over the text model alone. If none qualifies, links are not used.
    Metrics are then reported on the email test split.
    """
    t_text = json.loads(
        (params.resolve(Path("models/phishing/bundle")) / "phishing.json").read_text()
    )["threshold"]
    val, test = _email_frame(params, model, "val"), _email_frame(params, model, "test")
    yv = val["label"].to_numpy()
    text_v = val["prob"].to_numpy() >= t_text
    budget = text_v[yv == 0].mean() + params.phishing.url_email_extra_fpr
    candidates = np.unique(np.concatenate([np.linspace(0.5, 0.999, 60), [0.9995, 0.9999]]))
    chosen = float("inf")
    for t in candidates:  # ascending: first feasible threshold catches the most
        flag = text_v | (val["max_url_prob"].to_numpy() >= t)
        if flag[yv == 0].mean() <= budget:
            chosen = float(t)
            break
    text_t = test["prob"].to_numpy() >= t_text
    url_t = test["max_url_prob"].to_numpy() >= chosen
    return {
        "learned_fusion": learned_fusion(val, test, seed=params.seed),
        "emails": len(test),
        "with_urls": int((test["urls"].map(len) > 0).sum()),
        "urls_scored": test.attrs["urls_scored"],
        "email_url_threshold": chosen if np.isfinite(chosen) else None,
        "extra_fpr_budget": params.phishing.url_email_extra_fpr,
        "text_only": _verdict_metrics(test, text_t),
        "url_only": _verdict_metrics(test, url_t),
        "text_or_url": _verdict_metrics(test, text_t | url_t),
        # Same rule with the per-URL threshold, i.e. without validation tuning.
        "text_or_url_untuned": _verdict_metrics(
            test, text_t | (test["max_url_prob"].to_numpy() >= model.threshold)
        ),
    }


def _row_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df.groupby(["source", "label"]).size()
    sources = counts.index.get_level_values("source")
    labels = counts.index.get_level_values("label")
    return {
        f"{s}_{'malicious' if lab else 'legitimate'}": int(n)
        for s, lab, n in zip(sources, labels, counts.to_numpy(), strict=True)
    }


def train(params: Params, register: bool = True) -> dict[str, Any]:
    p = params.phishing
    df = split(load_urls(params.resolve(p.url_raw_dir)), params.seed)
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    log.info("URLs: %s", {s: len(v) for s, v in parts.items()})

    val, test = parts["val"], parts["test"]
    # Per-URL threshold from legitimate Hannousse validation URLs: they have paths and
    # queries like real links, whereas PhiUSIIL's legitimate URLs are bare homepages.
    realistic_legit = ((val["source"] == "Hannousse") & (val["label"] == 0)).to_numpy()

    def by_source(m: UrlModel, t: float) -> dict[str, dict[str, float]]:
        tp = m.predict_proba(test["url"].tolist())
        return {
            str(s): metrics(g["label"].to_numpy(), tp[(test["source"] == s).to_numpy()], t)
            for s, g in test.groupby("source")
        }

    unweighted = fit(parts["train"], val, p.url_max_features, params.seed, balance=False)
    uw_t = threshold_at_budget(
        unweighted.predict_proba(val["url"].tolist())[realistic_legit], p.url_fpr_budget
    )
    unweighted.threshold = uw_t
    t0 = time.perf_counter()
    model = fit(parts["train"], val, p.url_max_features, params.seed, balance=True)
    secs = time.perf_counter() - t0
    val_p = model.predict_proba(val["url"].tolist())
    model.threshold = threshold_at_budget(val_p[realistic_legit], p.url_fpr_budget)
    test_p = model.predict_proba(test["url"].tolist())
    results: dict[str, Any] = {
        "rows": {s: len(v) for s, v in parts.items()},
        "by_source_rows": _row_counts(df),
        "threshold": model.threshold,
        "fpr_budget": p.url_fpr_budget,
        "train_seconds": secs,
        "test": metrics(test["label"].to_numpy(), test_p, model.threshold),
        "test_by_source": by_source(model, model.threshold),
        "unweighted": {
            "test": metrics(
                test["label"].to_numpy(), unweighted.predict_proba(test["url"].tolist()), uw_t
            ),
            "test_by_source": by_source(unweighted, uw_t),
            "email_study": email_study(params, unweighted),
        },
    }

    # Baseline: logistic regression on the same character n-grams.
    Xtr = model.vectorizer.transform(parts["train"]["url"].map(normalize))
    lr = LogisticRegression(C=4.0, max_iter=2000).fit(Xtr, parts["train"]["label"])
    lr_val = lr.predict_proba(model.vectorizer.transform(parts["val"]["url"].map(normalize)))[:, 1]
    lr_t = threshold_at_budget(lr_val[parts["val"]["label"].to_numpy() == 0], p.url_fpr_budget)
    lr_test = lr.predict_proba(model.vectorizer.transform(test["url"].map(normalize)))[:, 1]
    results["baseline_logreg_tfidf"] = metrics(test["label"].to_numpy(), lr_test, lr_t)

    # Cross-dataset: PhiUSIIL only -> all of Hannousse.
    phi = df[df["source"] == "PhiUSIIL"]
    cross = fit(
        phi[phi["split"] == "train"], phi[phi["split"] == "val"], p.url_max_features, params.seed
    )
    cv = phi[phi["split"] == "val"]
    cross_t = threshold_at_budget(
        cross.predict_proba(cv.loc[cv["label"] == 0, "url"].tolist()), p.url_fpr_budget
    )
    han = df[df["source"] == "Hannousse"]
    results["cross_dataset_phiusiil_to_hannousse"] = metrics(
        han["label"].to_numpy(), cross.predict_proba(han["url"].tolist()), cross_t
    )

    results["email_study"] = email_study(params, model)
    model.email_threshold = results["email_study"]["email_url_threshold"]
    model.metadata = {
        "datasets": ["PhiUSIIL (UCI 967)", "Hannousse & Yahiouche 2021"],
        "rows": results["rows"],
    }
    bundle = params.resolve(Path("models/phishing/url_bundle"))
    model.save(bundle)

    reports = params.resolve(params.data.reports_dir) / "phishing"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "url_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    from sentinel.phishing.report import url_results_markdown

    (reports / "url_results.md").write_text(url_results_markdown(results), encoding="utf-8")
    (reports / "url_metrics.json").write_text(
        json.dumps(results["test"], indent=2), encoding="utf-8"
    )

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    decision = None
    with mlflow.start_run(run_name="url-train"):
        mlflow.log_params(
            {
                "sample": "full",
                "fpr_budget": p.url_fpr_budget,
                "max_features": p.url_max_features,
                "rounds": model.booster.best_iteration,
                **LGB_PARAMS,
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in results["test"].items()})
        for s, m in results["test_by_source"].items():
            mlflow.log_metrics({f"test_{s}_{k}": v for k, v in m.items()})
        mlflow.log_metrics(
            {f"cross_{k}": v for k, v in results["cross_dataset_phiusiil_to_hannousse"].items()}
        )
        mlflow.log_artifacts(str(reports), artifact_path="reports")
        from sentinel.phishing.registry import log_url_bundle, promote_url

        version = log_url_bundle(bundle, register=register)
        if version:
            decision = promote_url(version, 2 * p.url_fpr_budget)
    results["registry"] = decision
    return results
