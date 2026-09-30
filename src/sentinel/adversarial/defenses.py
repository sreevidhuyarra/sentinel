"""Defenses: adversarial training, feature hardening, disagreement flag."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sentinel.adversarial.attacks import CalibratedDetector, ZSpace, masked_pgd
from sentinel.common.config import AdversarialParams, LightGBMParams
from sentinel.common.logging import get_logger
from sentinel.ids.calibration import TemperatureBias
from sentinel.ids.dataset import BENIGN, CLASSES, Splits
from sentinel.ids.decision import choose_threshold
from sentinel.ids.models.gbm import fit_lightgbm
from sentinel.ids.models.mlp import MLPModel
from sentinel.ids.weights import class_weights, sample_weights

log = get_logger(__name__)


class ClampedNet(torch.nn.Module):
    """The MLP exactly as MLPModel.logits runs it: inputs clipped to +/-20 std first."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(torch.clamp(z, -20.0, 20.0))
        return out


def calibrated(
    name: str, model: Any, s: Splits, fpr_target: float, columns: np.ndarray | None = None
) -> CalibratedDetector:
    """Calibrate on validation and pick the benign-FPR-constrained threshold, as Module 1 does."""
    Xv = s.val.X if columns is None else s.val.X[:, columns]
    logits = model.logits(Xv)
    cal = TemperatureBias.fit(logits, s.val.y)
    choice = choose_threshold(cal.apply(logits), s.val.y, fpr_target)
    return CalibratedDetector(name, model, cal, choice.threshold, columns)


def _subsample(y: np.ndarray, per_class: int, benign: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(y == c)
        cap = benign if c == BENIGN else per_class
        keep.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    return np.sort(np.concatenate(keep))


def adversarial_training(
    mlp: MLPModel,
    s: Splits,
    zs: ZSpace,
    ctrl: np.ndarray,
    inc: np.ndarray,
    p: AdversarialParams,
    fpr_target: float,
    cap: float,
    seed: int,
) -> tuple[CalibratedDetector, list[dict[str, float]]]:
    """Fine-tune a copy of the deployed MLP on 50% clean + 50% PGD loss. Only attack flows
    are perturbed: benign traffic has no reason to be adversarial."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = _subsample(s.train.y, p.adv_train_per_class, p.adv_train_benign, seed)
    Z = torch.from_numpy(zs.to_z(s.train.X[idx]))
    Y = torch.from_numpy(s.train.y[idx])
    w = torch.tensor(class_weights(s.train.y[idx], 0.5, cap), dtype=torch.float32)
    w = w / w[Y].mean()
    model = copy.deepcopy(mlp)
    net = ClampedNet(model.net)
    opt = torch.optim.AdamW(net.parameters(), lr=p.adv_train_lr, weight_decay=1e-4)
    history = []
    for epoch in range(p.adv_train_epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(len(Z)))
        total = 0.0
        for i in range(0, len(Z), 2048):
            b = order[i : i + 2048]
            if len(b) < 2:
                continue
            zb, yb = Z[b], Y[b]
            attack = yb != BENIGN
            z_adv = zb.clone()
            if attack.any():
                z_adv[attack] = masked_pgd(
                    net, zb[attack], p.adv_train_eps, p.adv_train_steps, zs, ctrl, inc
                )
            loss = 0.5 * torch.nn.functional.cross_entropy(
                net(zb), yb, weight=w
            ) + 0.5 * torch.nn.functional.cross_entropy(net(z_adv), yb, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            total += loss.item() * len(b)
        history.append({"epoch": epoch, "loss": total / len(Z)})
        log.info("adversarial training epoch %d loss %.4f", epoch, total / len(Z))
    model.net.eval()  # ClampedNet wraps model.net, so the fine-tuned weights are already there
    return calibrated("mlp_adv_trained", model, s, fpr_target), history


def hardened_lightgbm(
    name: str,
    s: Splits,
    drop: set[str],
    lp: LightGBMParams,
    power: float,
    cap: float,
    fpr_target: float,
    seed: int,
) -> CalibratedDetector:
    """LightGBM retrained without the features in `drop`."""
    cols = np.array([i for i, c in enumerate(s.feature_names) if c not in drop])
    sw = sample_weights(s.train.y, power, cap)
    model = fit_lightgbm(
        s.train.X[:, cols],
        s.train.y,
        s.val.X[:, cols],
        s.val.y,
        lp,
        sw,
        seed,
        [s.feature_names[i] for i in cols],
    )
    return calibrated(name, model, s, fpr_target, columns=cols)


@dataclass
class DisagreementFlag:
    """Send a flow to human review when the two supervised members disagree strongly or
    the autoencoder cannot reconstruct it."""

    delta: float
    ae_threshold: float

    @classmethod
    def fit(
        cls,
        a: CalibratedDetector,
        b: CalibratedDetector,
        X_benign_val: np.ndarray,
        ae_threshold: float,
        budget: float,
    ) -> DisagreementFlag:
        gap = np.abs(a.attack_prob(X_benign_val) - b.attack_prob(X_benign_val))
        return cls(float(np.quantile(gap, 1 - budget, method="higher")), ae_threshold)

    def flags(
        self, a: CalibratedDetector, b: CalibratedDetector, ae: Any, X: np.ndarray
    ) -> dict[str, np.ndarray]:
        gap = np.abs(a.attack_prob(X) - b.attack_prob(X)) > self.delta
        anomalous = ae.score(X) >= self.ae_threshold
        return {"disagreement": gap, "anomaly": anomalous, "either": gap | anomalous}
