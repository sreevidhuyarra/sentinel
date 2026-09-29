"""PyTorch MLP with focal loss, the deep-learning member of the ensemble."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sentinel.common.config import MLPParams
from sentinel.common.logging import get_logger
from sentinel.ids.dataset import CLASSES
from sentinel.ids.metrics import macro_f1

log = get_logger(__name__)

_CLIP = 20.0
_EVAL_BATCH = 65_536


@dataclass
class Scaler:
    """Mean / std scaling, clipped at ±20 std.

    Inputs are already log-compressed by FeatureSpec, so outliers are tame. Median / IQR
    scaling was tried and lost ~5 points of macro-F1: 18 of 84 features are zero for
    over 75% of flows, so their IQR is 0 and the rare non-zero values carry the signal.
    """

    center: list[float]
    scale: list[float]

    @classmethod
    def fit(cls, X: np.ndarray) -> Scaler:
        mean = X.mean(axis=0, dtype=np.float64)
        std = X.std(axis=0, dtype=np.float64)
        return cls(center=mean.tolist(), scale=np.where(std > 1e-9, std, 1.0).tolist())

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = (X - np.asarray(self.center, dtype=np.float32)) / np.asarray(
            self.scale, dtype=np.float32
        )
        return np.clip(z, -_CLIP, _CLIP).astype(np.float32)


class Net(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout: float, n_out: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.body(x)
        return out


def focal_loss(
    logits: torch.Tensor, y: torch.Tensor, gamma: float, alpha: torch.Tensor | None = None
) -> torch.Tensor:
    """Multiclass focal loss (Lin et al. 2017): down-weights examples the model already
    classifies confidently, so the many easy benign flows stop dominating the gradient."""
    logp = torch.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
    loss = -((1 - logp.exp()) ** gamma) * logp
    if alpha is not None:
        loss = loss * alpha[y]
    return loss.mean()


@dataclass
class MLPModel:
    scaler: Scaler
    hidden: list[int]
    dropout: float
    n_features: int
    net: Net = field(repr=False)
    history: list[dict[str, float]] = field(default_factory=list)

    @torch.no_grad()
    def logits(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        Z = self.scaler.transform(X)
        outs = [
            self.net(torch.from_numpy(Z[i : i + _EVAL_BATCH]))
            for i in range(0, len(Z), _EVAL_BATCH)
        ]
        return torch.cat(outs).double().numpy() if outs else np.zeros((0, len(CLASSES)))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / "mlp.pt")
        meta = {
            "scaler": asdict(self.scaler),
            "hidden": self.hidden,
            "dropout": self.dropout,
            "n_features": self.n_features,
        }
        (directory / "mlp.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> MLPModel:
        meta = json.loads((directory / "mlp.json").read_text(encoding="utf-8"))
        net = Net(meta["n_features"], meta["hidden"], meta["dropout"], len(CLASSES))
        net.load_state_dict(torch.load(directory / "mlp.pt", weights_only=True))
        net.eval()
        return cls(
            Scaler(**meta["scaler"]), meta["hidden"], meta["dropout"], meta["n_features"], net
        )


def fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    p: MLPParams,
    class_weight: np.ndarray | None = None,
    seed: int = 42,
) -> MLPModel:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = Scaler.fit(X)
    Z = torch.from_numpy(scaler.transform(X))
    Y = torch.from_numpy(y)
    net = Net(X.shape[1], p.hidden, p.dropout, len(CLASSES))
    model = MLPModel(scaler, p.hidden, p.dropout, X.shape[1], net)

    alpha = None
    if class_weight is not None:
        # Normalise so the average training example has weight 1 (keeps the LR meaningful).
        w = torch.tensor(class_weight, dtype=torch.float32)
        alpha = w / w[Y].mean()

    steps = int(np.ceil(len(Z) / p.batch_size))
    opt = torch.optim.AdamW(net.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p.epochs * steps)

    best, best_state, stale = -1.0, copy.deepcopy(net.state_dict()), 0
    for epoch in range(p.epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(len(Z)))
        total = 0.0
        for i in range(steps):
            idx = order[i * p.batch_size : (i + 1) * p.batch_size]
            if len(idx) < 2:  # BatchNorm needs >1 row
                continue
            loss = focal_loss(net(Z[idx]), Y[idx], p.focal_gamma, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        f1 = macro_f1(y_val, model.logits(X_val).argmax(axis=1))
        model.history.append({"epoch": epoch, "train_loss": total / len(Z), "val_macro_f1": f1})
        log.info("mlp epoch %d loss %.4f val macro-F1 %.4f", epoch, total / len(Z), f1)
        if f1 > best:
            best, best_state, stale = f1, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= p.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return model
