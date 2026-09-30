"""Evasion attacks against the deployed detectors.

All attacks are judged by each detector's *deployed* decision rule (calibration + alert
threshold): an attack succeeds when a flow the detector caught is now called Benign.
Feature-space attacks work in standardised units z = (x - mean) / std of the
FeatureSpec-transformed features, so one budget unit means one training standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import polars as pl
import torch

from sentinel.adversarial.threat import pad_and_delay, project
from sentinel.data.features import FeatureSpec
from sentinel.ids.calibration import TemperatureBias
from sentinel.ids.dataset import BENIGN, CLASSES
from sentinel.ids.decision import decide


class Detector(Protocol):
    name: str

    def proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class CalibratedDetector:
    """A logit model + its calibration + its alert threshold, on a subset of columns."""

    name: str
    model: Any  # has .logits(X)
    calibration: TemperatureBias
    threshold: float
    columns: np.ndarray | None = None  # indices into the full feature list; None = all

    def proba(self, X: np.ndarray) -> np.ndarray:
        Xs = X if self.columns is None else X[:, self.columns]
        return self.calibration.apply(self.model.logits(Xs))

    def attack_prob(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(1.0 - self.proba(X)[:, BENIGN])

    def evaded(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(decide(self.proba(X), self.threshold) == BENIGN)


@dataclass
class EnsembleDetector:
    name: str
    bundle: Any  # IDSBundle

    def proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.bundle.proba(X))

    def attack_prob(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(1.0 - self.proba(X)[:, BENIGN])

    def evaded(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(decide(self.proba(X), self.bundle.threshold) == BENIGN)


@dataclass
class ZSpace:
    """Standardised coordinates shared by every attack (the MLP's own scaler)."""

    center: np.ndarray
    scale: np.ndarray
    lo: np.ndarray  # training min, in z
    hi: np.ndarray  # training max, in z

    def to_z(self, X: np.ndarray) -> np.ndarray:
        return np.asarray((X - self.center) / self.scale, dtype=np.float32)

    def to_x(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z * self.scale + self.center, dtype=np.float32)


def benign_target(n: int) -> np.ndarray:
    y = np.zeros((n, len(CLASSES)), dtype=np.float32)
    y[:, BENIGN] = 1.0
    return y


def _art_classifier(net: torch.nn.Module, zs: ZSpace) -> Any:
    from art.estimators.classification import PyTorchClassifier

    return PyTorchClassifier(
        model=net.eval(),
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(len(zs.center),),
        nb_classes=len(CLASSES),
        clip_values=(zs.lo.astype(np.float32), zs.hi.astype(np.float32)),
        device_type="cpu",
    )


def art_gradient_attack(
    method: str,
    net: torch.nn.Module,
    z: np.ndarray,
    eps: float,
    zs: ZSpace,
    ctrl: np.ndarray,
    inc: np.ndarray,
    max_iter: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Targeted (towards Benign) L-inf FGSM or PGD from ART, restricted to controllable
    features, then projected to a valid flow."""
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

    clf = _art_classifier(net, zs)
    mask = np.broadcast_to(ctrl.astype(np.float32), z.shape).copy()
    if method == "fgsm":
        attack = FastGradientMethod(clf, norm=np.inf, eps=eps, targeted=True, batch_size=512)
    elif method == "pgd":
        torch.manual_seed(seed)
        attack = ProjectedGradientDescent(
            clf,
            norm=np.inf,
            eps=eps,
            eps_step=eps / 4,
            max_iter=max_iter,
            targeted=True,
            num_random_init=0,
            batch_size=512,
            verbose=False,
        )
    else:
        raise ValueError(method)
    z_adv = attack.generate(z, y=benign_target(len(z)), mask=mask)
    return project(z_adv, z, zs.lo, zs.hi, ctrl, inc)


def masked_pgd(
    net: torch.nn.Module,
    z: torch.Tensor,
    eps: float,
    steps: int,
    zs: ZSpace,
    ctrl: np.ndarray,
    inc: np.ndarray,
) -> torch.Tensor:
    """Same threat model as `art_gradient_attack`, in plain torch, fast enough to generate
    adversarial examples inside every training batch (adversarial training)."""
    lo, hi = torch.from_numpy(zs.lo.astype(np.float32)), torch.from_numpy(zs.hi.astype(np.float32))
    c, i = torch.from_numpy(ctrl), torch.from_numpy(inc)
    target = torch.full((len(z),), BENIGN, dtype=torch.long)
    alpha = 2.5 * eps / steps
    was_training = net.training
    net.eval()
    x = z.clone()
    for _ in range(steps):
        x.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(net(x), target)
        (grad,) = torch.autograd.grad(loss, x)
        with torch.no_grad():
            x = x - alpha * grad.sign()  # descend the loss of the Benign target
            x = torch.clamp(x, z - eps, z + eps)
            x = torch.where(c, x, z)
            x = torch.where(i, torch.maximum(x, z), x)
            x = torch.clamp(x, lo, hi)
    net.train(was_training)
    return x.detach()


def hop_skip_jump(
    detector: CalibratedDetector,
    z: np.ndarray,
    donors: np.ndarray,
    zs: ZSpace,
    ctrl: np.ndarray,
    inc: np.ndarray,
    max_iter: int = 10,
    max_eval: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Decision-based black-box attack (HopSkipJump, ART) on the deployed decision rule.

    Two classes for the attack: 0 = scored Benign, 1 = alert. HopSkipJump needs a starting
    point that already evades; each flow gets its controllable features copied from a
    benign donor flow. Returns (z_adv, has_start); flows with no evading start keep z.
    """
    from art.attacks.evasion import HopSkipJump
    from art.estimators.classification import BlackBoxClassifier

    def predict(zb: np.ndarray) -> np.ndarray:
        ev = detector.evaded(zs.to_x(zb.astype(np.float32)))
        out = np.zeros((len(zb), 2), dtype=np.float32)
        out[ev, 0] = 1.0
        out[~ev, 1] = 1.0
        return out

    rng = np.random.default_rng(seed)
    init = z.copy()
    has_start = np.zeros(len(z), dtype=bool)
    for j in range(len(z)):
        for d in rng.choice(len(donors), size=min(20, len(donors)), replace=False):
            cand = project(np.where(ctrl, donors[d], z[j]), z[j], zs.lo, zs.hi, ctrl, inc)
            if predict(cand[None])[0, 0] == 1.0:
                init[j], has_start[j] = cand, True
                break
    clf = BlackBoxClassifier(
        predict,
        input_shape=(z.shape[1],),
        nb_classes=2,
        clip_values=(zs.lo.astype(np.float32), zs.hi.astype(np.float32)),
    )
    attack = HopSkipJump(
        clf,
        targeted=True,
        norm=np.inf,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=min(50, max_eval // 2),
        verbose=False,
    )
    idx = np.flatnonzero(has_start)
    z_adv = z.copy()
    if len(idx):
        mask = np.broadcast_to(ctrl.astype(np.float32), (len(idx), z.shape[1])).copy()
        y = np.zeros((len(idx), 2), dtype=np.float32)
        y[:, 0] = 1.0
        out = attack.generate(z[idx], y=y, x_adv_init=init[idx], mask=mask)
        z_adv[idx] = project(out, z[idx], zs.lo, zs.hi, ctrl, inc)
    return z_adv, has_start


def linf(z_adv: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(z_adv - z).max(axis=1))


@dataclass
class ProblemSpaceResult:
    evaded: dict[str, dict[str, np.ndarray]] = field(
        default_factory=dict
    )  # budget -> detector -> mask
    knob: dict[str, dict[str, np.ndarray]] = field(
        default_factory=dict
    )  # smallest (pad, delay) that works


def problem_space_search(
    frame: pl.DataFrame,
    spec: FeatureSpec,
    detectors: list[Any],
    pads: list[float],
    delays: list[float],
    budgets: dict[str, tuple[float, float]],
) -> ProblemSpaceResult:
    """Black-box search over padding x delay. A flow evades a detector within a budget if
    any (pad, delay) inside the budget makes that detector score it Benign."""
    grid = [(p, d) for p in pads for d in delays]
    hits = {}
    for p, d in grid:
        X = spec.to_numpy(pad_and_delay(frame, p, d))
        hits[(p, d)] = {det.name: det.evaded(X) for det in detectors}
    res = ProblemSpaceResult()
    for bname, (pmax, dmax) in budgets.items():
        inside = [(p, d) for p, d in grid if p <= pmax + 1e-9 and d <= dmax + 1e-9]
        res.evaded[bname] = {
            det.name: np.logical_or.reduce([hits[g][det.name] for g in inside]) for det in detectors
        }
    return res
