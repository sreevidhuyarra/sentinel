"""The whole Module 4 study end to end on the synthetic models (tiny budgets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentinel.adversarial.run import run
from sentinel.common.config import AdversarialParams


def test_robustness_study_end_to_end(trained_anomaly: dict[str, Any]) -> None:
    root: Path = trained_anomaly["root"]
    params = trained_anomaly["params"].model_copy(
        update={
            "adversarial": AdversarialParams(
                per_family=15,
                eps=[0.5],
                pgd_iter=3,
                hsj_samples=8,
                hsj_max_iter=2,
                hsj_max_eval=40,
                pads=[0.0, 0.5],
                delays=[1.0, 3.0],
                budgets={"low": (0.5, 3.0)},
                adv_train_epochs=1,
                adv_train_per_class=300,
                adv_train_benign=600,
            )
        }
    )
    out = run(
        params,
        ids_dir=root / "bundle",
        anomaly_dir=root / "anomaly",
        ids_results=root / "reports" / "results.json",
        out_dir=root / "adversarial",
        reports=root / "adversarial_reports",
        log_mlflow=False,
    )
    assert out["targets"]["total"] > 0
    assert set(out["clean"]) == {
        "lightgbm",
        "mlp",
        "ensemble",
        "mlp_adv_trained",
        "lightgbm_no_controllable",
        "lightgbm_no_timing",
    }
    for key in ("fgsm|mlp|0.5", "pgd|mlp|0.5", "pgd|mlp_adv_trained|0.5"):
        for v in out["feature_space"][key].values():
            assert v is None or 0.0 <= v <= 1.0
    assert set(out["problem_space"]) == {"low"}
    assert 0.0 <= out["hop_skip_jump"]["success_any_distance"] <= 1.0
    for name in ("results.md", "results.json", "metrics.json", "figures/evasion_vs_budget.png"):
        assert (root / "adversarial_reports" / name).exists(), name
    assert (root / "adversarial" / "mlp_adv_trained" / "mlp.pt").exists()
