"""Per-alert and global explanations from LightGBM TreeSHAP contributions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.ids.dataset import CLASSES

_WORDS = {
    "dst": "destination",
    "src": "source",
    "fwd": "forward",
    "bwd": "backward",
    "iat": "inter-arrival time",
    "pkts": "packets",
    "pkt": "packet",
    "psh": "PSH",
    "urg": "URG",
    "rst": "RST",
    "syn": "SYN",
    "ack": "ACK",
    "fin": "FIN",
    "cwr": "CWR",
    "ece": "ECE",
    "icmp": "ICMP",
    "tcp": "TCP",
    "avg": "average",
    "std": "std-dev",
    "min": "min",
    "max": "max",
    "init": "initial",
    "win": "window",
    "seg": "segment",
}
_UNITS = {"flow_duration": "µs", "total_tcp_flow_time": "µs"}


def display_name(feature: str) -> str:
    """'fwd_iat_mean' -> 'forward inter-arrival time mean'; rates end with 'per second'."""
    rate = feature.endswith("_s")
    stem = feature[:-2] if rate else feature
    words = [_WORDS.get(w, w) for w in stem.split("_")]
    name = " ".join(words)
    if "iat" in stem.split("_"):
        name += " (µs)"
    elif feature in _UNITS:
        name += f" ({_UNITS[feature]})"
    return name + (" per second" if rate else "")


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.3g}" if abs(value) < 1e4 else f"{value:,.0f}"


def reason_text(feature: str, value: float, contribution: float, family: str) -> str:
    direction = "increased" if contribution > 0 else "decreased"
    return f"{display_name(feature)} = {_fmt(value)} {direction} the likelihood of {family}"


def top_reasons(
    contrib: np.ndarray,
    raw_values: np.ndarray,
    feature_names: list[str],
    class_idx: int,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Top-k features pushing one flow toward `class_idx`.

    contrib: (n_classes, n_features + 1) TreeSHAP row; raw_values: untransformed features.
    Only positive contributions are reasons *for* the alert; if fewer than k exist, the
    largest-magnitude remaining ones fill the list so the analyst always sees k factors.
    """
    c = contrib[class_idx, :-1]
    order = np.argsort(-c)
    pos = [i for i in order if c[i] > 0][:k]
    if len(pos) < k:
        rest = [i for i in np.argsort(-np.abs(c)) if i not in pos]
        pos += rest[: k - len(pos)]
    family = CLASSES[class_idx]
    return [
        {
            "feature": feature_names[i],
            "value": float(raw_values[i]),
            "contribution": float(c[i]),
            "text": reason_text(feature_names[i], float(raw_values[i]), float(c[i]), family),
        }
        for i in pos
    ]


def global_importance(contrib: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Mean |SHAP| per feature for each family, computed on that family's own flows
    and that family's output — i.e. what the model looks at to recognise it."""
    out: dict[str, np.ndarray] = {}
    for i, c in enumerate(CLASSES):
        m = y == i
        if m.any():
            out[c] = np.abs(contrib[m, i, :-1]).mean(axis=0)
    return out


def plot_global_importance(
    importance: dict[str, np.ndarray], feature_names: list[str], out_dir: Path, top: int = 15
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for family, imp in importance.items():
        order = np.argsort(imp)[-top:]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([display_name(feature_names[i]) for i in order], imp[order], color="#2b7a8c")
        ax.set_xlabel("mean |SHAP value| (log-odds)")
        ax.set_title(f"What drives {family} predictions")
        fig.tight_layout()
        path = out_dir / f"shap_{re.sub(r'\W+', '_', family.lower())}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths
