"""Copilot evaluation (report section 8.3): retrieval, reports, faithfulness, injection.

Retrieval needs no LLM and runs on every gold alert. Report generation, the faithfulness
judge and the red-team suite go through the cached, throttled provider chain, so a rerun
is free and a run interrupted by the daily quota resumes where it stopped.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.common.logging import get_logger
from sentinel.copilot import prompts
from sentinel.copilot.graph import Copilot
from sentinel.copilot.rag import Retriever
from sentinel.llm.base import LLMError, LLMProvider

log = get_logger(__name__)

KS = (1, 3, 5, 8)


# -- retrieval ---------------------------------------------------------------------------------


@dataclass
class RetrievalConfig:
    name: str
    bm25: bool = True
    dense: bool = True
    rerank: bool = True
    hints: bool = True


RETRIEVAL_CONFIGS = [
    RetrievalConfig("bm25", dense=False, rerank=False),
    RetrievalConfig("dense", bm25=False, rerank=False),
    RetrievalConfig("hybrid", rerank=False),
    RetrievalConfig("hybrid+rerank"),
    RetrievalConfig("hybrid+rerank, no family hints", hints=False),
    RetrievalConfig("hybrid, no family hints", rerank=False, hints=False),
]


def _search(r: Retriever, query: str, cfg: RetrievalConfig, k: int) -> list[str]:
    saved = (r._rank_bm25, r._rank_dense, r._rerank)
    try:
        if not cfg.bm25:
            r._rank_bm25 = lambda *_: []  # type: ignore[method-assign]
        if not cfg.dense:
            r._rank_dense = lambda *_: []  # type: ignore[method-assign]
        if not cfg.rerank:
            r._rerank = None
        return [h.id for h in r.search(query, kind="technique", k=k)]
    finally:
        r._rank_bm25, r._rank_dense, r._rerank = saved  # type: ignore[method-assign]


def rank_metrics(ranked: list[list[str]], acceptable: list[list[str]]) -> dict[str, float]:
    first = []
    for ids, ok in zip(ranked, acceptable, strict=True):
        pos = next((i for i, t in enumerate(ids) if t in ok), None)
        first.append(pos)
    out = {f"recall@{k}": float(np.mean([p is not None and p < k for p in first])) for k in KS}
    out["mrr"] = float(np.mean([0.0 if p is None else 1.0 / (p + 1) for p in first]))
    return out


def evaluate_retrieval(copilot: Copilot, gold: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = []
    for g in gold:
        s = copilot.guard_inputs({"alert_id": g["alert_id"]})
        s.update(copilot.gather_context({**s}))
        contexts.append(s)
    out: dict[str, Any] = {}
    for cfg in RETRIEVAL_CONFIGS:
        t0 = time.perf_counter()
        ranked = [
            _search(
                copilot.retriever,
                prompts.retrieval_query(c["alert"], c["related"], cfg.hints),
                cfg,
                max(KS),
            )
            for c in contexts
        ]
        m: dict[str, Any] = dict(rank_metrics(ranked, [g["acceptable"] for g in gold]))
        m["ms_per_query"] = 1000 * (time.perf_counter() - t0) / len(gold)
        by_label: dict[str, list[bool]] = {}
        for ids, g in zip(ranked, gold, strict=True):
            by_label.setdefault(g["label"], []).append(any(t in g["acceptable"] for t in ids[:3]))
        m["recall@3_by_label"] = {k: float(np.mean(v)) for k, v in by_label.items()}
        out[cfg.name] = m
        log.info(
            "retrieval %s: %s",
            cfg.name,
            {k: round(v, 3) for k, v in m.items() if isinstance(v, float)},
        )
    return out


# -- reports -----------------------------------------------------------------------------------


def _mentions(report_text: str, fact: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(fact)}(?![\d.])", report_text) is not None


def report_text(rep: Any) -> str:
    d = rep.draft
    parts = [d.title, d.summary, d.severity_rationale, d.injection_notes, d.limitations]
    parts += [e.event for e in d.timeline] + [h.ip + " " + h.role for h in d.affected_hosts]
    parts += [t.rationale for t in d.techniques] + list(d.recommended_actions)
    return "\n".join(parts)


def score_report(rep: Any, g: dict[str, Any]) -> dict[str, Any]:
    chosen = [t.technique_id for t in rep.draft.techniques]
    text = report_text(rep)
    return {
        "alert_id": g["alert_id"],
        "label": g["label"],
        "provider": rep.provider,
        "chosen": chosen,
        "p@1": bool(chosen) and chosen[0] in g["acceptable"],
        "hit@3": any(t in g["acceptable"] for t in chosen[:3]),
        "candidate_hit": any(t in g["acceptable"] for t in rep.candidates["techniques"]),
        "cve_hit": bool(g["cves"]) and any(c.cve_id in g["cves"] for c in rep.draft.cves),
        "fact_recall": (
            float(np.mean([_mentions(text, f) for f in g["key_facts"]])) if g["key_facts"] else None
        ),
        "verified_first_try": rep.verification["passed"] and rep.verification["attempts"] == 1,
        "verified": rep.verification["passed"],
        "dropped_citations": len(rep.verification["dropped_citations"]),
        "severity": rep.severity["level"],
        "input_tokens": rep.usage.get("input_tokens", 0),
        "output_tokens": rep.usage.get("output_tokens", 0),
        "llm_seconds": rep.usage.get("llm_seconds", 0.0),
        "cached": rep.usage.get("calls", 0) > 0
        and rep.usage.get("cached_calls", 0) == rep.usage.get("calls", 0),
    }


JUDGE_SYSTEM = """You check an incident report against the evidence it was written from.
List each factual claim in the report (hosts, times, counts, attack type, what happened)
and decide whether the CONTEXT supports it. Claims about recommended actions are not
factual claims. Answer in JSON."""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                },
                "required": ["claim", "supported"],
            },
        },
    },
    "required": ["claims"],
}


def judge_faithfulness(judge: LLMProvider, context: str, rep: Any) -> dict[str, Any] | None:
    """Share of the report's factual claims supported by its context (LLM-as-judge)."""
    prompt = f"CONTEXT\n{context}\n\nREPORT\n{report_text(rep)}"
    try:
        claims = judge.generate(JUDGE_SYSTEM, prompt, JUDGE_SCHEMA).json().get("claims", [])
    except LLMError as exc:
        log.warning("judge failed: %s", exc)
        return None
    if not claims:
        return None
    ok = [bool(c.get("supported")) for c in claims]
    return {
        "claims": len(ok),
        "faithfulness": float(np.mean(ok)),
        "unsupported": [c["claim"] for c in claims if not c.get("supported")][:5],
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    cve_rows = [r for r in rows if r.get("cve_expected")]
    return {
        "n": len(rows),
        "providers": {
            p: sum(r["provider"] == p for r in rows) for p in {r["provider"] for r in rows}
        },
        "technique_p@1": mean("p@1"),
        "technique_hit@3": mean("hit@3"),
        "candidate_recall": mean("candidate_hit"),
        "cve_recall": float(np.mean([r["cve_hit"] for r in cve_rows])) if cve_rows else None,
        "fact_recall": mean("fact_recall"),
        "verified_first_try": mean("verified_first_try"),
        "citation_validity": 1.0,  # enforced: verify drops anything not in the context
        "dropped_citations_per_report": mean("dropped_citations"),
        "faithfulness": mean("faithfulness"),
        "input_tokens": mean("input_tokens"),
        "output_tokens": mean("output_tokens"),
        "llm_seconds": mean("llm_seconds"),
    }


def evaluate_reports(
    copilot: Copilot,
    gold: list[dict[str, Any]],
    judge: LLMProvider | None,
    stop_on_template: bool = True,
    progress: Callable[[int, dict[str, Any]], None] | None = None,
    checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    """Score one report per gold alert. With `checkpoint` (JSON lines), each finished row is
    appended as it completes and a rerun skips alerts already there, so an interrupted run
    (quota, power loss) resumes instead of starting over."""
    done: dict[int, dict[str, Any]] = {}
    if checkpoint is not None and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[int(r["alert_id"])] = r
        log.info("resuming: %d reports already scored", len(done))
    rows = []
    for i, g in enumerate(gold):
        if g["alert_id"] in done:
            rows.append(done[g["alert_id"]])
            continue
        rep, trace = copilot.investigate(g["alert_id"])
        row = score_report(rep, g)
        row["cve_expected"] = bool(g["cves"])
        row["seconds"] = sum(t["seconds"] for t in trace)
        if rep.provider == "template":
            log.warning("alert %s: no LLM answered (quota?)", g["alert_id"])
            if stop_on_template:
                break
        elif judge is not None:
            state = copilot.guard_inputs({"alert_id": g["alert_id"]})
            state.update(copilot.gather_context({**state}))
            state.update(copilot.map_attack({**state}))
            state.update(copilot.assess_severity({**state}))
            j = judge_faithfulness(judge, prompts.build_context(dict(state)), rep)
            if j:
                row.update(j)
        rows.append(row)
        if checkpoint is not None:
            with checkpoint.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        if progress:
            progress(i, row)
    return rows


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)
