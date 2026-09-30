"""Human-readable Module 3 results and model card."""

from __future__ import annotations

from typing import Any

_NAMES = {
    "pytorch_fp32": "PyTorch, float32 (reference)",
    "onnx_fp32": "ONNX, float32",
    "onnx_int8_embeddings": "ONNX, int8 embeddings only",
    "onnx_int8_full": "ONNX, int8 everything",
}


def _pct(v: float) -> str:
    return f"{v:.2%}"


def results_markdown(r: dict[str, Any]) -> str:
    t, h = r["accuracy"]["test"], r["accuracy_at_0.5"]["test"]
    lines = [
        "# Phishing email classifier results\n",
        "DeBERTa-v3-small fine-tuned with LoRA on Kaggle "
        f"({r['training']['minutes']:.0f} min on {', '.join(r['training']['gpus'])}). Label 1 = phishing, "
        "fraud or spam; 0 = legitimate. Full-split numbers come from the Kaggle run's predictions.\n",
        f"Alert threshold {r['threshold']:.3f}: the score exceeded by {r['fpr_budget']:.0%} of legitimate "
        "validation emails.\n",
        "## Test split (36,087 emails)\n",
        "| Threshold | F1 | Precision | Recall | False alarms on legitimate | Phishing caught | Fraud | Spam |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in (("0.5 (as trained)", h), (f"{r['threshold']:.3f} (deployed)", t)):
        k = m["flagged_by_kind"]
        lines.append(
            f"| {name} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{_pct(m['false_alarm_rate'])} | {_pct(k['phishing'])} | {_pct(k['fraud'])} | {_pct(k['spam'])} |"
        )
    lines += [
        f"\nThreshold-free: PR-AUC {t['pr_auc']:.4f}, ROC-AUC {t['roc_auc']:.4f}.\n",
        "## Per source corpus (test, deployed threshold)\n",
        "| Corpus | Emails | False alarms on legitimate | Malicious caught |",
        "|---|---|---|---|",
        *[
            f"| {s} | {v['emails']:,} | {_pct(v['false_alarm_rate']) if v['false_alarm_rate'] is not None else '-'} "
            f"| {_pct(v['recall']) if v['recall'] is not None else '-'} |"
            for s, v in t["by_source"].items()
        ],
        f"\n## Before and after optimisation ({r['benchmark_sample']:,} test emails, laptop CPU)\n",
        "| Model | Size | F1 | ROC-AUC | Verdicts differing from PyTorch | Latency p50 / p95 (1 email) | Emails/s (batched) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, b in r["benchmark"].items():
        size = "-" if b["size_mb"] is None else f"{b['size_mb']:.0f} MB"
        differ = b.get("verdicts_differ_from_pytorch", "-")
        mark = " **(deployed)**" if key == r["deployed"] else ""
        lines.append(
            f"| {_NAMES.get(key, key)}{mark} | {size} | {b['f1']:.4f} | {b['roc_auc']:.4f} | {differ} | "
            f"{b['p50_ms']:.0f} / {b['p95_ms']:.0f} ms | {b['emails_per_s']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def url_results_markdown(r: dict[str, Any]) -> str:
    def row(name: str, m: dict[str, float]) -> str:
        return (
            f"| {name} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['f1']:.4f} | "
            f"{_pct(m['false_alarm_rate'])} | {_pct(m['recall'])} |"
        )

    def verdict_rows(e: dict[str, Any], rows: list[tuple[str, str]]) -> list[str]:
        out = []
        for name, key in rows:
            m, k = e[key], e[key]["caught_by_kind"]
            out.append(
                f"| {name} | {_pct(m['false_alarm_rate'])} | {_pct(m['recall'])} | {m['f1']:.4f} | "
                f"{_pct(k.get('phishing', 0))} | {_pct(k.get('fraud', 0))} | {_pct(k.get('spam', 0))} |"
            )
        return out

    e, uw = r["email_study"], r["unweighted"]
    t_email = e["email_url_threshold"]
    t_text = f"{t_email:.4f}" if t_email is not None else "none qualified, links not used"
    lines = [
        "# URL classifier results\n",
        "Character 3-5-gram TF-IDF on normalised URLs + 23 lexical features -> LightGBM. Trained on "
        f"PhiUSIIL + Hannousse ({r['rows']['train']:,} train / {r['rows']['val']:,} validation / "
        f"{r['rows']['test']:,} test URLs), split by registered domain, each dataset given equal total "
        f"weight. Per-URL threshold {r['threshold']:.3f}: at most {r['fpr_budget']:.0%} of legitimate "
        "Hannousse validation URLs flagged (they have paths like real links; PhiUSIIL's legitimate URLs "
        "are all bare homepages).\n",
        "| Test set | ROC-AUC | PR-AUC | F1 | False alarms on legitimate | Malicious caught |",
        "|---|---|---|---|---|---|",
        row("**LightGBM, source-balanced (deployed)**", r["test"]),
        *[row(f"  of which {s}", m) for s, m in r["test_by_source"].items()],
        row("LightGBM, unweighted (first attempt)", uw["test"]),
        *[row(f"  of which {s}", m) for s, m in uw["test_by_source"].items()],
        row("Logistic regression on n-grams (baseline)", r["baseline_logreg_tfidf"]),
        row(
            "Cross-dataset: trained on PhiUSIIL only, scored on all Hannousse",
            r["cross_dataset_phiusiil_to_hannousse"],
        ),
        "\n## Combined verdict on the email test split\n",
        f"{e['emails']:,} test emails, {e['with_urls']:,} with at least one URL; {e['urls_scored']:,} "
        f"URLs scored (first {MAX_URLS} per email). A link flags its email when its score reaches the "
        "email threshold, chosen on the email validation split so links add at most "
        f"{e['extra_fpr_budget']:.1%} of false alarms: {t_text}.\n",
        "| Verdict | False alarms on legitimate | Malicious caught | F1 | Phishing | Fraud | Spam |",
        "|---|---|---|---|---|---|---|",
        *verdict_rows(
            e,
            [
                ("Text model alone", "text_only"),
                ("Links alone (email threshold)", "url_only"),
                ("**Text OR link (deployed)**", "text_or_url"),
                ("Text OR link at the per-URL threshold (untuned)", "text_or_url_untuned"),
            ],
        ),
        *verdict_rows(
            uw["email_study"],
            [
                ("Unweighted URL model: text OR link, untuned", "text_or_url_untuned"),
            ],
        ),
    ]
    fusion = e.get("learned_fusion")
    if fusion:
        w = fusion["weights"]
        lines += [
            "\n## Learned combination (fairer than OR)\n",
            "Logistic regression over the text logit, the best link's logit and has-link, fitted on half "
            "of the validation emails; thresholds for it and for text alone set on the other half at each "
            f"false-alarm budget, then scored on test. Weights: text {w['text_logit']:.2f}, "
            f"link {w['link_logit']:.2f}, has-link {w['has_link']:.2f}.\n",
            "| Validation budget | Method | Test false alarms | Malicious caught | Phishing caught |",
            "|---|---|---|---|---|",
            *[
                f"| {x['budget']:.1%} | {x['method']} | {_pct(x['false_alarm_rate'])} | "
                f"{_pct(x['recall'])} | {_pct(x['phishing'])} |"
                for x in fusion["rows"]
            ],
        ]
    return "\n".join(lines) + "\n"


MAX_URLS = 20


def model_card(r: dict[str, Any]) -> str:
    t = r["accuracy"]["test"]
    b = r["benchmark"][r["deployed"]]
    return f"""# Model card: phishing-classifier

## Intended use
Score an email (subject + body) for phishing, fraud or spam so a SOC analyst can triage it.
It flags for review; it does not quarantine or delete mail on its own.

## Model
DeBERTa-v3-small (141M parameters) fine-tuned with LoRA (rank 16 on the attention
query/key/value projections; classification head and pooler trained in full; 0.72% of
weights trainable). Merged and exported to ONNX; {r["deployed"].replace("_", " ")} deployed
({b["size_mb"]:.0f} MB). Input: subject + body, HTML removed, URLs replaced by [URL],
first 256 tokens. Alert when P(malicious) >= {r["threshold"]:.3f}.

## Data
Phishing Email Curated Datasets (Champa, Rabbi & Zibran 2024; Zenodo 8339691; CC BY 4.0):
11 public corpora, 200,517 emails after deduplication, split by sender domain / subject
campaign so no sender or campaign appears in two splits.

## Results (test split, deployed threshold)
F1 {t["f1"]:.4f}, precision {t["precision"]:.4f}, recall {t["recall"]:.4f}, false alarms on
legitimate mail {_pct(t["false_alarm_rate"])}; PR-AUC {t["pr_auc"]:.4f}. Phishing caught
{_pct(t["flagged_by_kind"]["phishing"])}, fraud {_pct(t["flagged_by_kind"]["fraud"])}, spam
{_pct(t["flagged_by_kind"]["spam"])}.

## Limitations
- **Mostly spam, little phishing.** Of ~89k malicious training emails only ~900 are true
  phishing (Nazario corpus); phishing recall is the weakest per-kind number.
- **Old mail.** The corpora span 1995-2022, much of it 2002-2008. Modern phishing
  (QR codes, MFA-fatigue lures, LLM-written text) is under-represented.
- **Corpus effects.** False alarms concentrate in some corpora (TREC), whose "legitimate"
  mail includes newsletters and bulk mail; the model partly learns corpus style.
- **Text only.** URLs are replaced by [URL]; link reputation needs the separate URL model.
- **Threshold trade-off.** At 0.5 it catches more ({_pct(r["accuracy_at_0.5"]["test"]["recall"])}) but
  flags {_pct(r["accuracy_at_0.5"]["test"]["false_alarm_rate"])} of legitimate mail.
"""
