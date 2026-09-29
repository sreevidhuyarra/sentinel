import numpy as np
import pytest
from sklearn.metrics import f1_score

from sentinel.ids.calibration import TemperatureBias
from sentinel.ids.dataset import BENIGN, CLASSES
from sentinel.ids.decision import attack_score, choose_threshold, decide, severity
from sentinel.ids.explain import display_name, top_reasons
from sentinel.ids.metrics import benign_fpr, evaluate, expected_calibration_error, macro_f1
from sentinel.ids.weights import class_weights

K = len(CLASSES)
rng = np.random.default_rng(0)


def test_fast_macro_f1_matches_sklearn() -> None:
    y = rng.integers(0, K, 5000)
    pred = np.where(rng.random(5000) < 0.7, y, rng.integers(0, K, 5000))
    expected = f1_score(y, pred, labels=range(K), average="macro", zero_division=0)
    assert macro_f1(y, pred) == pytest.approx(expected)


def test_benign_fpr() -> None:
    y = np.array([BENIGN] * 4 + [1])
    pred = np.array([BENIGN, BENIGN, BENIGN, 1, 1])
    assert benign_fpr(y, pred) == 0.25


def test_ece_zero_for_perfect_confident_predictions() -> None:
    y = rng.integers(0, K, 1000)
    proba = np.eye(K)[y]
    assert expected_calibration_error(y, proba) == pytest.approx(0.0)


def test_evaluate_reports_all_classes() -> None:
    y = np.arange(K).repeat(10)
    m = evaluate(y, y, np.eye(K)[y])
    assert m["macro_f1"] == 1.0 and set(m["per_class"]) == set(CLASSES)


def test_class_weights() -> None:
    y = np.array([0] * 1000 + [1] * 10 + [2] * 1)
    assert np.allclose(class_weights(y, power=0.0)[:3], 1.0)
    w = class_weights(y, power=1.0, cap=500)
    assert w[1] == pytest.approx(100) and w[2] == 500  # capped
    assert class_weights(y, power=0.5)[1] == pytest.approx(10)


def _proba(y: np.ndarray, noise: float = 0.3) -> np.ndarray:
    logits = np.eye(K)[y] * 4 + rng.normal(0, noise * 4, (len(y), K))
    return TemperatureBias.identity(K).apply(logits)


def test_decide_threshold_controls_attacks() -> None:
    p = np.full((2, K), 0.0)
    p[0, BENIGN], p[0, 3] = 0.7, 0.3  # attack score 0.3
    p[1, BENIGN], p[1, 5] = 0.2, 0.8
    assert decide(p, 0.5).tolist() == [BENIGN, 5]
    assert decide(p, 0.25).tolist() == [3, 5]
    assert attack_score(p).tolist() == pytest.approx([0.3, 0.8])


def test_choose_threshold_respects_fpr_budget() -> None:
    y = np.concatenate([np.full(5000, BENIGN), rng.integers(1, K, 1000)])
    p = _proba(y, noise=0.5)
    choice = choose_threshold(p, y, fpr_target=0.01)
    assert choice.val_benign_fpr <= 0.01
    assert benign_fpr(y, decide(p, choice.threshold)) <= 0.01


def test_severity_bands() -> None:
    assert severity(np.array([0.1, 0.6, 0.85, 0.99]), [0.6, 0.8, 0.95]) == [
        "Low",
        "Medium",
        "High",
        "Critical",
    ]


def test_temperature_bias_recovers_scaling_and_prior() -> None:
    # Labels drawn from softmax(true_logits), so true_logits are calibrated by construction.
    true_logits = rng.normal(0, 2, (20000, K))
    p_true = TemperatureBias.identity(K).apply(true_logits)
    y = np.array([rng.choice(K, p=row) for row in p_true])
    over = true_logits * 4  # overconfident by exactly 4x
    cal = TemperatureBias.fit(over, y)
    assert 3.5 < cal.temperature < 4.5
    assert np.abs(cal.bias).max() < 0.1
    p = cal.apply(over)
    assert np.allclose(p.sum(axis=1), 1)
    assert expected_calibration_error(y, p) < expected_calibration_error(
        y, TemperatureBias.identity(K).apply(over)
    )


def test_display_names() -> None:
    assert display_name("dst_port") == "destination port"
    assert display_name("flow_bytes_s") == "flow bytes per second"
    assert display_name("fwd_iat_mean") == "forward inter-arrival time mean (µs)"


def test_top_reasons_positive_first() -> None:
    contrib = np.zeros((K, 4))
    contrib[2, :3] = [0.5, -2.0, 1.5]
    reasons = top_reasons(contrib, np.array([22, 3, 180]), ["dst_port", "a", "b"], 2, k=2)
    assert [r["feature"] for r in reasons] == ["b", "dst_port"]
    assert "increased the likelihood of" in reasons[0]["text"]
