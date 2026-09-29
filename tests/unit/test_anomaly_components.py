import numpy as np
import pytest

from sentinel.anomaly.autoencoder import AENet, fit_autoencoder
from sentinel.anomaly.bundle import _unlog
from sentinel.anomaly.evaluate import detection, recall_at_fprs, threshold_at_fpr
from sentinel.anomaly.iforest import fit_iforest
from sentinel.common.config import AutoencoderParams, IForestParams
from sentinel.data.features import FeatureSpec

rng = np.random.default_rng(0)


def _low_rank(n: int, d: int = 20, rank: int = 3) -> np.ndarray:
    basis = rng.normal(0, 1, (rank, d))
    return (rng.normal(0, 1, (n, rank)) @ basis + rng.normal(0, 0.05, (n, d))).astype(np.float32)


def test_aenet_shapes() -> None:
    import torch

    net = AENet(20, [16, 8], 4)
    assert net(torch.zeros(5, 20)).shape == (5, 20)
    assert net.encoder(torch.zeros(5, 20)).shape == (5, 4)


def test_autoencoder_scores_off_manifold_points_higher() -> None:
    normal = _low_rank(4000)
    p = AutoencoderParams(hidden=[16, 8], epochs=40, patience=5, batch_size=256, lr=3e-3)
    model = fit_autoencoder(normal[:3000], normal[3000:], p, bottleneck=3, seed=0)
    outliers = rng.normal(0, 2, (200, 20)).astype(np.float32)
    s_norm, s_out = model.score(normal[3000:]), model.score(outliers)
    assert np.median(s_out) > 5 * np.median(s_norm)
    labels = np.array(["Benign"] * 1000 + ["X"] * 200)
    d = detection(np.concatenate([s_norm, s_out]), threshold_at_fpr(s_norm, 0.01), labels)
    assert d["roc_auc"] > 0.95 and d["attack_recall"] > 0.9


def test_iforest_scores_outliers_higher() -> None:
    normal = rng.normal(0, 1, (3000, 5))
    model = fit_iforest(normal, IForestParams(n_estimators=50, train_rows=3000), seed=0)
    assert model.score(np.full((1, 5), 6.0))[0] > np.quantile(model.score(normal), 0.99)


def test_threshold_at_fpr() -> None:
    scores = np.arange(1000, dtype=float)
    t = threshold_at_fpr(scores, 0.01)
    assert (scores >= t).mean() == pytest.approx(0.01, abs=0.002)


def test_detection_per_label_and_sweep() -> None:
    score = np.array([0.1, 0.2, 0.3, 5.0, 6.0, 0.15])
    labels = np.array(["Benign", "Benign", "Benign", "Bot", "DoS", "DoS"])
    d = detection(score, 1.0, labels)
    assert d["benign_fpr"] == 0.0 and d["attack_recall"] == pytest.approx(2 / 3)
    assert d["per_label_recall"]["Bot"] == {"recall": 1.0, "flows": 1}
    assert d["per_label_recall"]["DoS"]["recall"] == 0.5
    sweep = recall_at_fprs(score[:3], score, labels, [0.01, 0.5])
    assert set(sweep) == {"0.01", "0.5"}


def test_unlog_inverts_signed_log1p() -> None:
    spec = FeatureSpec(columns=["a", "b"], log_columns=["a"])
    raw = np.array([[1000.0, 5.0], [-20.0, 7.0]])
    logged = raw.copy()
    logged[:, 0] = np.sign(raw[:, 0]) * np.log1p(np.abs(raw[:, 0]))
    assert np.allclose(_unlog(logged, spec), raw)
