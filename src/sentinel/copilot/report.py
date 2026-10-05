"""reports/copilot/results.md from whichever study outputs exist."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _pct(x: Any) -> str:
    return "n/a" if x is None else f"{100 * float(x):.1f}%"


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def guard_section(g: dict[str, Any]) -> list[str]:
    lines = [
        "## Injection guard",
        "",
        f"Trained on {g['data']['train_segments']:,} segments; chosen on validation: **{g['chosen']}** "
        f"(threshold {g['selection'][g['chosen']]['threshold']:.3f}, set for <= 1% false alarms on clean validation documents).",
        "",
        "| Candidate | Val recall | Val false alarms | Train time |",
        "|---|---|---|---|",
    ]
    for name, s in g["selection"].items():
        lines.append(
            f"| {name} | {_pct(s['val_recall'])} | {_pct(s['val_fpr'])} | {s['train_seconds']:.0f} s |"
        )
    lines += [
        "",
        "Held-out test (injection phrasings, email texts and hard negatives never seen in training):",
        "",
        "| Detector | Source | Recall | False alarms | Injections / clean |",
        "|---|---|---|---|---|",
    ]
    for det, by_src in g["test"].items():
        for src, m in by_src.items():
            lines.append(
                f"| {det} | {src} | {_pct(m['recall'])} | {_pct(m['fpr'])} | {m['n_pos']} / {m['n_neg']} |"
            )
    lines.append(f"\nScanning cost: {g['ms_per_document']:.0f} ms per document on CPU.")
    prev = g.get("previous")
    if prev:
        p, n = prev["test"]["rules+classifier"]["all"], g["test"]["rules+classifier"]["all"]
        lines.append(
            f"\nSentence-pair scoring (added after the red team found an injection split over two "
            f"sentences): threshold re-chosen on validation {prev['threshold']:.3f} -> "
            f"{g['selection'][g['chosen']]['threshold']:.3f}; test recall {_pct(p['recall'])} -> "
            f"{_pct(n['recall'])}, false alarms {_pct(p['fpr'])} -> {_pct(n['fpr'])}."
        )
    return lines


def retrieval_section(r: dict[str, Any], n: int) -> list[str]:
    lines = [
        "## Retrieval (no LLM)",
        "",
        f"{n} gold alerts. Recall@k = an acceptable technique is among the top k candidates; MRR = mean reciprocal rank of the first acceptable one.",
        "",
        "| Retriever | R@1 | R@3 | R@5 | R@8 | MRR | ms/query |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in r.items():
        lines.append(
            f"| {name} | {_pct(m['recall@1'])} | {_pct(m['recall@3'])} | {_pct(m['recall@5'])} | "
            f"{_pct(m['recall@8'])} | {m['mrr']:.3f} | {m['ms_per_query']:.0f} |"
        )
    best = r.get("hybrid+rerank")
    if best:
        lines += ["", "Recall@3 by label (hybrid + re-rank):", "", "| Label | R@3 |", "|---|---|"]
        lines += [f"| {k} | {_pct(v)} |" for k, v in sorted(best["recall@3_by_label"].items())]
    return lines


def reports_section(s: dict[str, Any]) -> list[str]:
    lines = [
        "## Incident reports (LLM)",
        "",
        f"{s['n']} reports; generator {s.get('generator')}, providers used {s.get('providers')}; judge {s.get('judge')}.",
        "",
        "| Metric | Value | Target (design) |",
        "|---|---|---|",
        f"| Technique precision@1 | {_pct(s['technique_p@1'])} | >= 70% |",
        f"| Technique hit@3 | {_pct(s['technique_hit@3'])} | |",
        f"| Acceptable technique among candidates | {_pct(s['candidate_recall'])} | |",
        f"| CVE recall (alerts with a known CVE) | {_pct(s['cve_recall'])} | |",
        f"| Key facts in the prose (IPs, port) | {_pct(s['fact_recall'])} | |",
        f"| Facts block complete (attacker, target; built by code) | {_pct(s.get('facts_block_complete'))} | |",
        f"| Reports still inventing an IP after verify | {_pct(s.get('reports_with_invented_ips'))} | 0% |",
        f"| Passed verification on the first draft | {_pct(s['verified_first_try'])} | |",
        f"| Citation validity (after verify) | {_pct(s['citation_validity'])} | 100% |",
        f"| Invalid citations dropped per report | {s['dropped_citations_per_report'] or 0:.2f} | |",
        f"| Faithfulness (judge: supported claims) | {_pct(s['faithfulness'])} | |",
        f"| Tokens in / out per report | {s['input_tokens'] or 0:,.0f} / {s['output_tokens'] or 0:,.0f} | |",
        f"| LLM seconds per report | {s['llm_seconds'] or 0:.1f} | |",
    ]
    if s.get("p@1_by_label"):
        lines += ["", "| Label | P@1 |", "|---|---|"]
        lines += [f"| {k} | {_pct(v)} |" for k, v in sorted(s["p@1_by_label"].items())]
    return lines


def redteam_section(rt: dict[str, Any]) -> list[str]:
    lines = [
        "## Red team: prompt injection",
        "",
        f"{len(rt['cases'])} attacked alerts (email bodies and correlated log lines carrying one "
        f"injection in a held-out phrasing); generator {rt.get('generator')}.",
        "",
        "| Defense | Cases | Attack success | canary | downgrade | leak | Guard detected |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in rt["summary"].items():
        g = s["by_goal"]
        lines.append(
            f"| {name} | {s['cases']} | {_pct(s['attack_success_rate'])} | {_pct(g.get('canary'))} | "
            f"{_pct(g.get('downgrade'))} | {_pct(g.get('leak'))} | {_pct(s.get('guard_detection_rate'))} |"
        )
    lines.append(
        "\nSeverity is rule-based in every configuration, so no injection can lower it; "
        "a detected injection raises it by one level."
    )
    return lines


def write_markdown(out_dir: Path) -> Path:
    lines = ["# SOC copilot results", ""]
    g = _load(out_dir / "guard_results.json")
    ev = _load(out_dir / "eval_results.json")
    rt = _load(out_dir / "redteam.json")
    if g:
        lines += [*guard_section(g), ""]
    if ev:
        lines += [*retrieval_section(ev["retrieval"], ev["gold_alerts"]), ""]
        if ev.get("reports"):
            lines += [*reports_section(ev["reports"]), ""]
    if rt:
        lines += [*redteam_section(rt), ""]
    path = out_dir / "results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
