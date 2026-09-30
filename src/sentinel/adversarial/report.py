"""Module 4 report: tables and the evasion-vs-budget figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentinel.common.config import AdversarialParams

NAMES = {
    "lightgbm": "LightGBM (deployed member)",
    "mlp": "MLP (deployed member)",
    "ensemble": "Ensemble (deployed)",
    "mlp_adv_trained": "MLP, adversarially trained",
    "lightgbm_no_controllable": "LightGBM without controllable features",
    "lightgbm_no_timing": "LightGBM without timing features",
    "ensemble_plus_review": "Ensemble + review flag",
}


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1%}"


def results_markdown(r: dict[str, Any], a: AdversarialParams) -> str:
    fs, eps = r["feature_space"], a.eps
    lines = [
        "# Adversarial robustness results\n",
        f"Targets: {r['targets']['total']:,} attack flows from the test split that the deployed ensemble "
        f"catches ({', '.join(f'{k} {v}' for k, v in r['targets']['per_family'].items())}). Evasion = share "
        "of the flows a detector caught that it scores Benign after the attack.\n",
        "The attacker may change only its own padding, timing and TCP window "
        f"({len(r['controllable_features'])} of 84 features; padding and delay only upwards), never "
        "destination port, protocol, flags, packet counts or the victim's replies.\n",
        "## Clean test accuracy (full test split)\n",
        "| Detector | Macro-F1 | Benign FPR | Attack recall |",
        "|---|---|---|---|",
        *[
            f"| {NAMES.get(n, n)} | {m['macro_f1']:.4f} | {m['benign_fpr']:.3%} | {m['attack_recall']:.4f} |"
            for n, m in r["clean"].items()
        ],
        f"\nReview flag (disagreement > {r['review_flag']['delta']:.3f} or autoencoder anomaly) sends "
        f"{_pct(r['review_flag']['benign_flag_rate_test']['either'])} of benign test flows to review "
        f"(disagreement {_pct(r['review_flag']['benign_flag_rate_test']['disagreement'])}, anomaly "
        f"{_pct(r['review_flag']['benign_flag_rate_test']['anomaly'])}).\n",
        "## Feature-space attacks (upper bound)\n",
        "Targeted towards Benign, L-inf budget in training standard deviations, ART FGSM / PGD "
        f"({a.pgd_iter} steps) with the controllable-feature mask and validity projection. White-box on the "
        "source MLP; every other column is a transfer attack.\n",
    ]
    cols = [
        "mlp",
        "lightgbm",
        "ensemble",
        "ensemble_plus_review",
        "mlp_adv_trained",
        "lightgbm_no_timing",
        "lightgbm_no_controllable",
    ]
    lines += [
        "| Attack | Source | Budget | " + " | ".join(NAMES[c] for c in cols) + " |",
        "|---|---|---|" + "---|" * len(cols),
    ]
    for method in ("fgsm", "pgd"):
        for source in ("mlp", "mlp_adv_trained"):
            for e in eps:
                row = fs.get(f"{method}|{source}|{e:g}")
                if row:
                    lines.append(
                        f"| {method.upper()} | {NAMES[source]} | {e:g} | "
                        + " | ".join(_pct(row.get(c)) for c in cols)
                        + " |"
                    )
    h = r["hop_skip_jump"]
    lines += [
        "\n## Black-box attack on LightGBM (HopSkipJump)\n",
        f"{h['flows']} flows; an evading starting point was found for {_pct(h['found_evading_start'])}. "
        f"Success at any distance: {_pct(h['success_any_distance'])}; median L-inf of successes "
        f"{h['median_linf_of_successes'] if h['median_linf_of_successes'] is None else round(h['median_linf_of_successes'], 3)}.\n",
        "| Budget | " + " | ".join(f"{e:g}" for e in eps) + " |",
        "|---|" + "---|" * len(eps),
        "| Success | " + " | ".join(_pct(h["success_at_eps"][f"{e:g}"]) for e in eps) + " |",
        "\n## Problem-space attacks (realistic)\n",
        "Every forward packet padded by up to `pad` of its payload and all timing stretched by up to `delay`, "
        "dependent features recomputed; the attacker tries every grid point inside the budget.\n",
        "| Detector | "
        + " | ".join(f"{b} (pad <= {v[0]:g}, delay <= {v[1]:g}x)" for b, v in a.budgets.items())
        + " |",
        "|---|" + "---|" * len(a.budgets),
    ]
    for name in [
        "lightgbm",
        "mlp",
        "ensemble",
        "ensemble_plus_review",
        "mlp_adv_trained",
        "lightgbm_no_timing",
        "lightgbm_no_controllable",
    ]:
        lines.append(
            f"| {NAMES[name]} | "
            + " | ".join(_pct(r["problem_space"][b].get(name)) for b in a.budgets)
            + " |"
        )
    lines += [
        f"\n### By family (deployed ensemble, {list(a.budgets)[-1]} budget)\n",
        "| Family | Evaded | Padding only | Delay only | Evaded despite review flag |",
        "|---|---|---|---|---|",
        *[
            f"| {f} | {_pct(v['ensemble_top'])} | {_pct(v['padding_only'])} | {_pct(v['delay_only'])} | "
            f"{_pct(v['ensemble_plus_review_top'])} |"
            for f, v in r["problem_space_by_family"].items()
        ],
    ]
    return "\n".join(lines) + "\n"


def plot_curves(r: dict[str, Any], eps: list[float], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fs = r["feature_space"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    series = [
        ("pgd|mlp", "mlp", "MLP, white-box PGD", "-o"),
        ("pgd|mlp_adv_trained", "mlp_adv_trained", "Adv.-trained MLP, white-box PGD", "-s"),
        ("pgd|mlp", "lightgbm", "LightGBM, transfer from MLP", "--^"),
        ("pgd|mlp", "ensemble", "Ensemble, transfer from MLP", "--d"),
        ("pgd|mlp", "ensemble_plus_review", "Ensemble + review flag, transfer", ":x"),
    ]
    for prefix, det, label, style in series:
        ys = [fs.get(f"{prefix}|{e:g}", {}).get(det) for e in eps]
        if any(v is not None for v in ys):
            ax.plot(eps, [100 * (v or 0) for v in ys], style, label=label, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("L-inf budget (training standard deviations)")
    ax.set_ylabel("evasion rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Feature-space evasion vs attacker budget")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
