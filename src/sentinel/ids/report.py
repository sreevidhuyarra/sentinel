"""Human-readable results: comparison table, per-class table, confusion-matrix figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sentinel.ids.dataset import CLASSES


def comparison_table(results: dict[str, dict[str, Any]]) -> str:
    head = (
        "| Model | Macro-F1 | Benign FPR | Attack recall | PR-AUC (macro) | ECE | Train (s) "
        "| Latency (ms / 1k flows) |\n|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name, r in results.items():
        m = r["test"]
        rows.append(
            f"| {name} | {m['macro_f1']:.4f} | {m['benign_fpr']:.4%} | {m['attack_recall']:.4f} "
            f"| {m['pr_auc_macro']:.4f} | {m['ece']:.4f} | {r['train_seconds']:.0f} "
            f"| {r['latency_ms_per_1k']:.1f} |"
        )
    return head + "\n".join(rows) + "\n"


def per_class_table(metrics: dict[str, Any]) -> str:
    out = "| Family | Precision | Recall | F1 | PR-AUC | Test flows |\n|---|---|---|---|---|---|\n"
    for c, m in metrics["per_class"].items():
        pr = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "-"
        out += (
            f"| {c} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {pr} "
            f"| {m['support']:,} |\n"
        )
    return out


def model_card(
    results: dict[str, dict[str, Any]],
    top_features: dict[str, list[str]],
    split_sizes: dict[str, int],
) -> str:
    ens = results["ensemble"]
    m = ens["test"]
    rare = [c for c, v in m["per_class"].items() if v["support"] < 100]
    drivers = "\n".join(
        f"| {fam} | {', '.join(f'`{f}`' for f in feats)} |" for fam, feats in top_features.items()
    )
    return f"""# Model card: ids-classifier

## Intended use
Classify network flows (CICFlowMeter features) as Benign or one of seven attack families
(DoS, DDoS, PortScan, BruteForce, WebAttack, Bot, Infiltration) to raise explained alerts
for a SOC analyst. It recommends; a human decides. Not for automated blocking.

## Model
Weighted average of a LightGBM classifier (weight {ens["lgbm_weight"]:.1f}) and a PyTorch MLP,
each calibrated with temperature + per-class bias on validation. A flow is an alert when
P(attack) >= {ens["threshold"]:.3f}, the threshold with the best validation macro-F1 at
benign FPR <= 1%. Explanations are TreeSHAP values from the LightGBM member.

## Data
Corrected CIC-IDS2017 (Engelen et al., DistriNet), 5 days of lab traffic, exact duplicates
removed. Split in time within every (day, label): train {split_sizes["train"]:,},
validation {split_sizes["val"]:,}, test {split_sizes["test"]:,} flows. "Attempted" attacks
(no payload delivered) are labelled Benign; "Infiltration - Portscan" is labelled PortScan.

## Results (test split)
{comparison_table(results)}
{per_class_table(m)}

## What the model relies on (top mean |SHAP| per family)
| Family | Top features |
|---|---|
{drivers}

## Limitations
- **One lab network; does not transfer.** All training traffic comes from one 2017
  testbed. Applied unchanged to UNSW-NB15 (`sentinel ids cross-dataset`), models trained
  here detect under 10% of attacks with ROC-AUC 0.55-0.68, while a model trained on
  UNSW-NB15 itself reaches 0.995 (reports/ids/cross_dataset.md). Retrain or fine-tune on
  the target network before use.
- **Possible testbed shortcuts.** TCP initial window sizes (`bwd_init_win_bytes`,
  `fwd_init_win_bytes`) and destination port carry much of the signal for some families;
  these can reflect the lab's specific hosts rather than attack behaviour. Module 4 tests
  models trained without them.
- **Tiny rare-class test sets.** {", ".join(rare) or "None"} have fewer than 100 test flows,
  so their recall can move by several points from one or two flows.
- **Known attacks only.** Families absent from training are not recognised; the anomaly
  detector (Module 2) covers novel behaviour.
- **Not adversarially hardened yet** (Module 4).
"""


def plot_confusion(cm: list[list[int]], path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(cm, dtype=np.float64)
    norm = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if arr[i, j]:
                color = "white" if norm[i, j] > 0.5 else "black"
                ax.text(
                    j, i, f"{int(arr[i, j]):,}", ha="center", va="center", fontsize=7, color=color
                )
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
