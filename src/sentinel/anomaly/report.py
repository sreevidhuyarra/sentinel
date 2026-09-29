"""Human-readable Module 2 results: tables, score-distribution figure, model card."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sentinel.ids.dataset import CLASSES


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.2%}"


def _auc(v: float | None) -> str:
    return "-" if v is None else f"{v:.4f}"


def results_markdown(r: dict[str, Any]) -> str:
    ae, iforest = r["autoencoder"], r["iforest"]
    lines = [
        "# Anomaly detection results (CIC-IDS2017 test split)\n",
        f"Both detectors are trained on benign training flows only. Thresholds are set on benign "
        f"validation flows. Autoencoder bottleneck: {r['bottleneck']} (chosen on validation).\n",
        "## Bottleneck sweep (validation)\n",
        "| Bottleneck | Val ROC-AUC | Epochs | Val benign MSE | Seconds |",
        "|---|---|---|---|---|",
        *[
            f"| {row['bottleneck']} | {row['val_roc_auc']:.4f} | {row['epochs']} | "
            f"{row['val_benign_mse']:.4f} | {row['seconds']:.0f} |"
            for row in r["bottleneck_sweep"]
        ],
        "\n## Benign vs attack (test)\n",
        "| Detector | ROC-AUC | PR-AUC | Benign FPR | Attack recall |",
        "|---|---|---|---|---|",
    ]
    for d in (ae, iforest):
        t = d["test"]
        lines.append(
            f"| {d['name']} | {_auc(t['roc_auc'])} | {_auc(t['pr_auc'])} | "
            f"{_pct(t['benign_fpr'])} | {_pct(t['attack_recall'])} |"
        )
    lines += [
        "\n## Recall per family at the configured threshold (test)\n",
        "| Family | Test flows | Autoencoder | Isolation Forest |",
        "|---|---|---|---|",
    ]
    for fam in CLASSES[1:]:
        a = ae["test"]["per_label_recall"].get(fam)
        b = iforest["test"]["per_label_recall"].get(fam)
        if a:
            lines.append(
                f"| {fam} | {a['flows']:,} | {_pct(a['recall'])} | "
                f"{_pct(b['recall'] if b else None)} |"
            )
    lines += [
        "\n## False-alarm budget trade-off (thresholds from benign validation, scored on test)\n",
        "| Target benign FPR | Autoencoder: benign FPR / attack recall "
        "| Isolation Forest: benign FPR / attack recall |",
        "|---|---|---|",
    ]
    for f, a in ae["fpr_sweep"].items():
        b = iforest["fpr_sweep"][f]
        lines.append(
            f"| {float(f):.1%} | {_pct(a['benign_fpr'])} / {_pct(a['attack_recall'])} | "
            f"{_pct(b['benign_fpr'])} / {_pct(b['attack_recall'])} |"
        )
    if "holdout" in r:
        lines += [
            "\n## Unknown-attack test: family removed from supervised training\n",
            "LightGBM is retrained without the family; all of that family's flows are then "
            "scored. Recall = share flagged as any kind of alert.\n",
            "| Held-out family | Flows | Supervised alone | Autoencoder alone "
            "| Isolation Forest alone "
            "| **Supervised + autoencoder** | Supervised + IF | Benign FPR sup. -> fused |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for fam, h in r["holdout"].items():
            lines.append(
                f"| {fam} | {h['flows']:,} | {_pct(h['supervised_recall'])} | "
                f"{_pct(h['autoencoder_recall'])} | {_pct(h['iforest_recall'])} | "
                f"**{_pct(h['fused_recall'])}** | {_pct(h['fused_iforest_recall'])} | "
                f"{_pct(h['supervised_benign_fpr'])} -> {_pct(h['fused_benign_fpr'])} |"
            )
    if "fusion" in r:
        f = r["fusion"]
        lines += [
            "\n## Deployed supervised model + autoencoder (all families known, test)\n",
            "| | Benign FPR | Benign false alarms | Attack recall |",
            "|---|---|---|---|",
            *[
                f"| {n} | {_pct(f[n]['benign_fpr'])} | {f[n]['benign_false_alarms']:,} | "
                f"{_pct(f[n]['attack_recall'])} |"
                for n in ("supervised", "fused")
            ],
            f"\n'Unknown anomaly' alerts added: {f['unknown_anomaly_alerts']['on_attacks']:,} on "
            f"attack flows, {f['unknown_anomaly_alerts']['on_benign']:,} on benign flows.",
        ]
    if "unsw" in r:
        u = r["unsw"]
        lines += [
            "\n## New network: CIC-UNSW-NB15 (no UNSW attack labels used)\n",
            f"{u['n_features']} shared features, {u['test_flows']:,} UNSW test flows. For "
            "comparison, the supervised models transfer at ROC-AUC 0.55-0.68, and a supervised "
            "model trained with UNSW attack labels reaches 0.995 (reports/ids/cross_dataset.md).\n",
            "| Detector | Needs from the new network | ROC-AUC | Benign FPR | Attack recall |",
            "|---|---|---|---|---|",
        ]
        rows = [
            ("CIC-trained autoencoder, as is", "nothing", u["cic_autoencoder_as_is"]),
            (
                "CIC-trained autoencoder, re-thresholded",
                "benign traffic",
                u["cic_autoencoder_rethresholded"],
            ),
            (
                "Autoencoder retrained on UNSW benign",
                "benign traffic",
                u["unsw_autoencoder_benign_only"],
            ),
            (
                "Isolation Forest trained on UNSW benign",
                "benign traffic",
                u["unsw_iforest_benign_only"],
            ),
        ]
        for name, need, d in rows:
            lines.append(
                f"| {name} | {need} | {_auc(d['roc_auc'])} | {_pct(d['benign_fpr'])} | "
                f"{_pct(d['attack_recall'])} |"
            )
        lines += [
            "\nRecall per UNSW category (autoencoder retrained on UNSW benign):\n",
            "| Category | Flows | Recall |",
            "|---|---|---|",
            *[
                f"| {c} | {v['flows']:,} | {_pct(v['recall'])} |"
                for c, v in sorted(
                    u["unsw_autoencoder_benign_only"]["per_label_recall"].items(),
                    key=lambda kv: -kv[1]["flows"],
                )
            ],
        ]
    return "\n".join(lines) + "\n"


def model_card(r: dict[str, Any], fpr_target: float) -> str:
    t = r["autoencoder"]["test"]
    return f"""# Model card: anomaly-detector

## Intended use
Flag network flows that look unlike normal traffic, including attack types absent from the
supervised model's training data. Used behind the supervised classifier: a flow it calls
Benign but the autoencoder cannot reconstruct becomes an "Unknown anomaly" alert (severity
Low, or Medium at twice the threshold) for an analyst to triage.

## Model
Autoencoder (84 -> 64 -> 32 -> {r["bottleneck"]} -> 32 -> 64 -> 84, ReLU, linear code) on
standard-scaled FeatureSpec features, trained with MSE on benign flows only. Anomaly score =
mean squared reconstruction error. Threshold = the score exceeded by {fpr_target:.0%} of benign
validation flows. Explanations list the worst-reconstructed features with the observed value
and the value the model reconstructs for a normal-looking flow.

## Data
Benign flows of the corrected CIC-IDS2017 training split. No attack labels are used for
training; validation attacks are used only to choose the bottleneck size.

## Results (CIC-IDS2017 test split)
ROC-AUC {_auc(t["roc_auc"])}, benign FPR {_pct(t["benign_fpr"])}, attack recall
{_pct(t["attack_recall"])}. See results.md for per-family recall, the unknown-attack test and
the new-network test.

## Limitations
- **False alarms are the price.** At a 1% benign FPR the detector adds roughly one alert per
  hundred benign flows; the supervised model alone raises about one per hundred thousand.
  Anomaly alerts are therefore kept at Low / Medium severity.
- **Unusual is not malicious.** Rare but legitimate traffic (new services, backups, scans by
  IT) scores high; slow or low-volume attacks that resemble normal flows score low.
- **Learns one network's normal.** Trained on CIC-IDS2017 benign traffic; on another network
  it must at least be re-thresholded, and preferably retrained, on that network's benign
  traffic (see the new-network table).
"""


def plot_scores(score: np.ndarray, labels: np.ndarray, threshold: float, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(max(score.min(), 1e-4)), np.log10(score.max() + 1e-9), 80).tolist()
    for fam in CLASSES:
        m = labels == fam
        if m.sum() >= 20:
            # Share of the family's flows per (log-spaced) bin, so families of very
            # different sizes are comparable; density=True would favour narrow bins.
            w = np.full(int(m.sum()), 1.0 / m.sum())
            ax.hist(score[m], bins=bins, weights=w, histtype="step", linewidth=1.4, label=fam)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label="threshold")
    ax.set_xscale("log")
    ax.set_xlabel("reconstruction error (log scale)")
    ax.set_ylabel("share of the family's flows")
    ax.set_title("Autoencoder anomaly score by family (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
