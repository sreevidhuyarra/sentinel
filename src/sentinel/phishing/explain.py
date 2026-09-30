"""Which parts of an email drove the phishing score: sentence occlusion.

Each sentence (or line) of the cleaned email is removed in turn and the email re-scored;
the drop in the malicious logit is that sentence's contribution. All variants are scored
in one batch through the ONNX model, so explanations need no PyTorch and no gradients.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from sentinel.phishing.export import PhishingOnnxModel

# The model reads ~256 tokens (~1,000-1,500 characters); text beyond that cannot matter.
MAX_CHARS = 1500
MAX_SEGMENTS = 24
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def segments(text: str) -> list[str]:
    parts = [s.strip() for s in _SPLIT.split(text[:MAX_CHARS]) if s and s.strip()]
    return parts[:MAX_SEGMENTS]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.asarray(np.log(p / (1 - p)))


def explain(model: PhishingOnnxModel, text: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Up to `top_k` sentences whose removal lowers the malicious score most."""
    segs = segments(text)
    if len(segs) < 2:
        return []
    variants = [" ".join(segs)] + [
        " ".join(s for j, s in enumerate(segs) if j != i) for i in range(len(segs))
    ]
    z = _logit(model.predict_proba(variants))
    contrib = z[0] - z[1:]
    order = np.argsort(-contrib)
    return [
        {"text": segs[i][:200], "contribution": float(contrib[i])}
        for i in order[:top_k]
        if contrib[i] > 0
    ]
