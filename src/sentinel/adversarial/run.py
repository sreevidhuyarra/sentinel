"""Module 4 study: attack the deployed detectors, apply defenses, measure both sides.

Targets are attack flows from the test split that the deployed ensemble catches (up to
`per_family` per family). A detector's evasion rate counts only targets that detector
caught before the attack. Every defense also reports its clean test accuracy, so the
robustness gained can be weighed against the accuracy lost.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import polars as pl

from sentinel.adversarial.attacks import (
    CalibratedDetector,
    EnsembleDetector,
    ZSpace,
    art_gradient_attack,
    hop_skip_jump,
    linf,
    problem_space_search,
)
from sentinel.adversarial.defenses import (
    ClampedNet,
    DisagreementFlag,
    adversarial_training,
    hardened_lightgbm,
)
from sentinel.adversarial.threat import CONTROLLABLE, TIMING, masks
from sentinel.anomaly.bundle import AnomalyBundle
from sentinel.common.config import Params, get_settings
from sentinel.common.logging import get_logger
from sentinel.ids.bundle import IDSBundle
from sentinel.ids.dataset import BENIGN, CLASSES, load_splits
from sentinel.ids.decision import decide
from sentinel.ids.metrics import evaluate

log = get_logger(__name__)

EXPERIMENT = "sentinel-adversarial"


@dataclass
class ReviewedEnsemble:
    """The ensemble plus the review flag: a flow evades only if the ensemble calls it
    Benign *and* the flag does not send it to an analyst."""

    name: str
    ensemble: EnsembleDetector
    flag: DisagreementFlag
    a: CalibratedDetector
    b: CalibratedDetector
    ae: Any

    def evaded(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.ensemble.evaded(X) & ~self.flag.flags(self.a, self.b, self.ae, X)["either"]
        )


def evasion_rate(det: Any, X_clean: np.ndarray, X_adv: np.ndarray) -> float | None:
    caught = ~det.evaded(X_clean)
    return float(det.evaded(X_adv)[caught].mean()) if caught.any() else None


def select_targets(
    test_frame: pl.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    ens: EnsembleDetector,
    per_family: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    caught = (y != BENIGN) & ~ens.evaded(X)
    picked = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(caught & (y == c))
        if len(idx):
            picked.append(
                idx if len(idx) <= per_family else rng.choice(idx, per_family, replace=False)
            )
    return np.sort(np.concatenate(picked))


def clean_metrics(det: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    m = evaluate(
        y,
        decide(det.proba(X), det.threshold if hasattr(det, "threshold") else det.bundle.threshold),
        det.proba(X),
    )
    return {k: m[k] for k in ("macro_f1", "benign_fpr", "attack_recall")}


def run(
    params: Params,
    ids_dir: Path | None = None,
    anomaly_dir: Path | None = None,
    ids_results: Path | None = None,
    out_dir: Path | None = None,
    reports: Path | None = None,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    a = params.adversarial
    t_start = time.perf_counter()
    reports = reports or params.resolve(params.data.reports_dir) / "adversarial"
    out_dir = out_dir or params.resolve(Path("models/adversarial"))
    reports.mkdir(parents=True, exist_ok=True)

    s = load_splits(params)
    bundle = IDSBundle.load(ids_dir or params.resolve(Path("models/ids")))
    anomaly = AnomalyBundle.load(anomaly_dir or params.resolve(Path("models/anomaly")))
    ids_results = ids_results or params.resolve(params.data.reports_dir) / "ids" / "results.json"
    thresholds = json.loads(ids_results.read_text())
    assert bundle.mlp is not None and bundle.mlp_calibration is not None
    lgbm = CalibratedDetector(
        "lightgbm", bundle.lgbm, bundle.lgbm_calibration, thresholds["lightgbm"]["threshold"]
    )
    mlp = CalibratedDetector(
        "mlp", bundle.mlp, bundle.mlp_calibration, thresholds["mlp"]["threshold"]
    )
    ens = EnsembleDetector("ensemble", bundle)

    ctrl, inc = masks(s.feature_names)
    sc = bundle.mlp.scaler
    zs = ZSpace(
        np.asarray(sc.center, dtype=np.float32),
        np.asarray(sc.scale, dtype=np.float32),
        np.zeros(1),
        np.zeros(1),
    )
    z_train = zs.to_z(s.train.X)
    zs.lo, zs.hi = z_train.min(axis=0), z_train.max(axis=0)
    del z_train

    Xt, yt = s.test.X, s.test.y
    tgt = select_targets(s.test.frame, Xt, yt, ens, a.per_family, params.seed)
    X0, y0 = Xt[tgt], yt[tgt]
    fam0 = np.array([CLASSES[c] for c in y0])
    log.info(
        "attack targets: %d flows %s",
        len(tgt),
        dict(zip(*np.unique(fam0, return_counts=True), strict=True)),
    )

    # Defenses (trained before any attack is run against them).
    t0 = time.perf_counter()
    mlp_adv, adv_history = adversarial_training(
        bundle.mlp,
        s,
        zs,
        ctrl,
        inc,
        a,
        params.ids.benign_fpr_target,
        params.ids.max_class_weight,
        params.seed,
    )
    adv_secs = time.perf_counter() - t0
    lp, power = (
        params.ids.lightgbm,
        params.ids.lightgbm.class_weight_power or params.ids.class_weight_power,
    )
    hard_all = hardened_lightgbm(
        "lightgbm_no_controllable",
        s,
        set(CONTROLLABLE),
        lp,
        power,
        params.ids.max_class_weight,
        params.ids.benign_fpr_target,
        params.seed,
    )
    hard_timing = hardened_lightgbm(
        "lightgbm_no_timing",
        s,
        set(TIMING),
        lp,
        power,
        params.ids.max_class_weight,
        params.ids.benign_fpr_target,
        params.seed,
    )
    flag = DisagreementFlag.fit(
        lgbm, mlp, s.val.X[s.val.y == BENIGN], anomaly.threshold, a.review_budget
    )
    reviewed = ReviewedEnsemble("ensemble_plus_review", ens, flag, lgbm, mlp, anomaly.autoencoder)
    detectors: list[Any] = [lgbm, mlp, ens, mlp_adv, hard_all, hard_timing]

    results: dict[str, Any] = {
        "targets": {
            "total": len(tgt),
            "per_family": {f: int((fam0 == f).sum()) for f in np.unique(fam0)},
        },
        "controllable_features": [c for c in s.feature_names if c in CONTROLLABLE],
        "clean": {d.name: clean_metrics(d, Xt, yt) for d in detectors},
        "adversarial_training": {"seconds": adv_secs, "history": adv_history},
        "review_flag": {"delta": flag.delta},
    }
    benign_test = Xt[yt == BENIGN]
    fl = flag.flags(lgbm, mlp, anomaly.autoencoder, benign_test)
    results["review_flag"]["benign_flag_rate_test"] = {k: float(v.mean()) for k, v in fl.items()}

    # Feature space: white-box on the MLPs, transfer to the rest.
    Z0 = zs.to_z(X0)
    feature_space: dict[str, Any] = {}
    for source_name, source in (("mlp", mlp), ("mlp_adv_trained", mlp_adv)):
        net = ClampedNet(source.model.net)
        for method in ("fgsm", "pgd"):
            for eps in a.eps:
                Z_adv = art_gradient_attack(
                    method, net, Z0, eps, zs, ctrl, inc, a.pgd_iter, params.seed
                )
                X_adv = (X0 + (Z_adv - Z0) * zs.scale).astype(np.float32)
                row = {d.name: evasion_rate(d, X0, X_adv) for d in [*detectors, reviewed]}
                feature_space[f"{method}|{source_name}|{eps:g}"] = row
                log.info(
                    "%s from %s eps=%g: %s",
                    method,
                    source_name,
                    eps,
                    {k: None if v is None else round(v, 3) for k, v in row.items()},
                )
    results["feature_space"] = feature_space

    # Black box: HopSkipJump on the deployed LightGBM decision.
    rng = np.random.default_rng(params.seed)
    per_fam = max(1, a.hsj_samples // len(np.unique(fam0)))
    hsj_idx = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(fam0 == f), min(per_fam, int((fam0 == f).sum())), replace=False
            )
            for f in np.unique(fam0)
        ]
    )
    benign_val = s.val.X[s.val.y == BENIGN]
    donors = zs.to_z(
        benign_val[rng.choice(len(benign_val), min(2000, len(benign_val)), replace=False)]
    )
    t0 = time.perf_counter()
    z_hsj, has_start = hop_skip_jump(
        lgbm, Z0[hsj_idx], donors, zs, ctrl, inc, a.hsj_max_iter, a.hsj_max_eval, params.seed
    )
    X_hsj = (X0[hsj_idx] + (z_hsj - Z0[hsj_idx]) * zs.scale).astype(np.float32)
    ev = lgbm.evaded(X_hsj)
    dist = linf(z_hsj, Z0[hsj_idx])
    results["hop_skip_jump"] = {
        "flows": len(hsj_idx),
        "found_evading_start": float(has_start.mean()),
        "seconds": time.perf_counter() - t0,
        "success_at_eps": {f"{e:g}": float((ev & (dist <= e + 1e-6)).mean()) for e in a.eps},
        "success_any_distance": float(ev.mean()),
        "median_linf_of_successes": float(np.median(dist[ev])) if ev.any() else None,
    }

    # Problem space: pad and delay, black-box search, every detector + review flag.
    frame0 = s.test.frame[tgt.tolist()]
    budgets = {k: (float(v[0]), float(v[1])) for k, v in a.budgets.items()}
    top = max(budgets, key=lambda k: budgets[k][0] + budgets[k][1])  # largest budget
    ps = problem_space_search(frame0, s.spec, [*detectors, reviewed], a.pads, a.delays, budgets)
    pad_only = problem_space_search(frame0, s.spec, [ens], a.pads, [1.0], {top: budgets[top]})
    delay_only = problem_space_search(frame0, s.spec, [ens], [0.0], a.delays, {top: budgets[top]})
    caught = {d.name: ~d.evaded(X0) for d in [*detectors, reviewed]}
    results["problem_space"] = {
        b: {n: float(m[caught[n]].mean()) if caught[n].any() else None for n, m in res.items()}
        for b, res in ps.evaded.items()
    }
    ens_high = ps.evaded[top]["ensemble"]
    results["problem_space_by_family"] = {
        f: {
            "ensemble_top": float(ens_high[fam0 == f].mean()),
            "padding_only": float(pad_only.evaded[top]["ensemble"][fam0 == f].mean()),
            "delay_only": float(delay_only.evaded[top]["ensemble"][fam0 == f].mean()),
            "ensemble_plus_review_top": float(
                ps.evaded[top]["ensemble_plus_review"][fam0 == f].mean()
            ),
        }
        for f in np.unique(fam0)
    }
    results["seconds"] = time.perf_counter() - t_start

    out_dir.mkdir(parents=True, exist_ok=True)
    mlp_adv.model.save(out_dir / "mlp_adv_trained")
    for det in (hard_all, hard_timing):
        det.model.save(out_dir / f"{det.name}.txt")
    (out_dir / "calibration.json").write_text(
        json.dumps(
            {
                d.name: {
                    "calibration": d.calibration.as_dict(),
                    "threshold": d.threshold,
                    "columns": None if d.columns is None else d.columns.tolist(),
                }
                for d in (mlp_adv, hard_all, hard_timing)
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    from sentinel.adversarial.report import plot_curves, results_markdown

    plot_curves(results, a.eps, reports / "figures" / "evasion_vs_budget.png")
    (reports / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (reports / "results.md").write_text(results_markdown(results, a), encoding="utf-8")
    headline = {
        "pgd_mlp_eps0.5": feature_space.get("pgd|mlp|0.5", {}).get("mlp"),
        "pgd_mlp_adv_eps0.5": feature_space.get("pgd|mlp_adv_trained|0.5", {}).get(
            "mlp_adv_trained"
        ),
        "problem_top_ensemble": results["problem_space"][top]["ensemble"],
        "problem_top_ensemble_review": results["problem_space"][top]["ensemble_plus_review"],
    }
    (reports / "metrics.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")

    if not log_mlflow:
        return results
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="robustness-study"):
        mlflow.log_params(
            {
                "targets": len(tgt),
                "adv_train_eps": a.adv_train_eps,
                "adv_train_steps": a.adv_train_steps,
                "adv_train_epochs": a.adv_train_epochs,
                "review_budget": a.review_budget,
            }
        )
        mlflow.log_metrics({k: v for k, v in headline.items() if v is not None})
        for name, m in results["clean"].items():
            mlflow.log_metrics({f"clean_{name}_{k}": v for k, v in m.items()})
        mlflow.log_artifacts(str(reports), artifact_path="reports")
    return results
