"""Autoencoder anomaly detector.

Trained to reconstruct benign flows only. A flow it reconstructs badly does not look
like anything in normal traffic, so the reconstruction error is the anomaly score. No
attack labels are used for training, which is what lets it flag attack types absent
from the supervised model's training data.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sentinel.common.config import AutoencoderParams
from sentinel.common.logging import get_logger
from sentinel.ids.models.mlp import Scaler

log = get_logger(__name__)

_EVAL_BATCH = 65_536


class AENet(nn.Module):
    """Symmetric MLP: n -> hidden... -> bottleneck -> ...hidden -> n."""

    def __init__(self, n: int, hidden: list[int], bottleneck: int) -> None:
        super().__init__()
        enc: list[nn.Module] = []
        prev = n
        for h in [*hidden, bottleneck]:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec: list[nn.Module] = []
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, n))
        self.encoder = nn.Sequential(*enc[:-1])  # linear code layer
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.decoder(self.encoder(x))
        return out


@dataclass
class AutoencoderModel:
    scaler: Scaler
    hidden: list[int]
    bottleneck: int
    n_features: int
    net: AENet = field(repr=False)
    history: list[dict[str, float]] = field(default_factory=list)

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(scaled input, scaled reconstruction), both (n, n_features)."""
        self.net.eval()
        Z = self.scaler.transform(X)
        outs = [
            self.net(torch.from_numpy(Z[i : i + _EVAL_BATCH])).numpy()
            for i in range(0, len(Z), _EVAL_BATCH)
        ]
        R = np.concatenate(outs) if outs else np.zeros_like(Z)
        return Z, R

    def feature_errors(self, X: np.ndarray) -> np.ndarray:
        Z, R = self.reconstruct(X)
        return np.asarray((Z - R) ** 2)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error per flow (in standard-scaled units)."""
        return np.asarray(self.feature_errors(X).mean(axis=1), dtype=np.float64)

    def expected(self, X: np.ndarray) -> np.ndarray:
        """Reconstruction mapped back to the FeatureSpec (log-compressed) scale."""
        _, R = self.reconstruct(X)
        return R * np.asarray(self.scaler.scale) + np.asarray(self.scaler.center)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / "autoencoder.pt")
        meta = {
            "scaler": asdict(self.scaler),
            "hidden": self.hidden,
            "bottleneck": self.bottleneck,
            "n_features": self.n_features,
        }
        (directory / "autoencoder.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> AutoencoderModel:
        meta = json.loads((directory / "autoencoder.json").read_text(encoding="utf-8"))
        net = AENet(meta["n_features"], meta["hidden"], meta["bottleneck"])
        net.load_state_dict(torch.load(directory / "autoencoder.pt", weights_only=True))
        net.eval()
        return cls(
            Scaler(**meta["scaler"]), meta["hidden"], meta["bottleneck"], meta["n_features"], net
        )


def fit_autoencoder(
    X: np.ndarray, X_val: np.ndarray, p: AutoencoderParams, bottleneck: int, seed: int = 42
) -> AutoencoderModel:
    """X, X_val: benign flows only. Early stopping on validation reconstruction MSE."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scaler = Scaler.fit(X)
    Z = torch.from_numpy(scaler.transform(X))
    Z_val = torch.from_numpy(scaler.transform(X_val))
    net = AENet(X.shape[1], p.hidden, bottleneck)
    model = AutoencoderModel(scaler, p.hidden, bottleneck, X.shape[1], net)
    opt = torch.optim.Adam(net.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    steps = int(np.ceil(len(Z) / p.batch_size))
    best, best_state, stale = np.inf, copy.deepcopy(net.state_dict()), 0
    for epoch in range(p.epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(len(Z)))
        total = 0.0
        for i in range(steps):
            batch = Z[order[i * p.batch_size : (i + 1) * p.batch_size]]
            loss = nn.functional.mse_loss(net(batch), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            total += loss.item() * len(batch)
        net.eval()
        with torch.no_grad():
            sq = sum(
                float(((net(Z_val[j : j + _EVAL_BATCH]) - Z_val[j : j + _EVAL_BATCH]) ** 2).sum())
                for j in range(0, len(Z_val), _EVAL_BATCH)
            )
        val = sq / Z_val.numel()
        model.history.append({"epoch": epoch, "train_mse": total / len(Z), "val_mse": val})
        log.info(
            "autoencoder k=%d epoch %d train %.4f val %.4f", bottleneck, epoch, total / len(Z), val
        )
        if val < best - 1e-5:
            best, best_state, stale = val, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= p.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return model
