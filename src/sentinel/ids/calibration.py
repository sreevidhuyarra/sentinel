"""Post-hoc probability calibration fitted on the validation split.

Plain temperature scaling fixes over/under-confidence but not the prior shift that
class-weighted training introduces (a weighted model believes rare attacks are far more
common than they are). Adding one bias per class to the scaled logits corrects both,
with only n_classes + 1 parameters, so it cannot overfit the validation set meaningfully.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class TemperatureBias:
    temperature: float
    bias: list[float]

    @classmethod
    def identity(cls, n_classes: int) -> TemperatureBias:
        return cls(1.0, [0.0] * n_classes)

    @classmethod
    def fit(cls, logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> TemperatureBias:
        L = torch.from_numpy(logits.astype(np.float64))
        Y = torch.from_numpy(y)
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(logits.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS(
            [log_t, b], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(L / log_t.exp() + b, Y)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        opt.step(closure)  # type: ignore[no-untyped-call]
        return cls(log_t.detach().exp().item(), b.detach().tolist())

    def apply(self, logits: np.ndarray) -> np.ndarray:
        z = logits / self.temperature + np.asarray(self.bias)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return np.asarray(e / e.sum(axis=1, keepdims=True))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
