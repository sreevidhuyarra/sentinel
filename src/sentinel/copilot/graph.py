"""The investigation state machine (report section 8.1).

    guard_inputs -> gather_context -> map_attack -> assess_severity -> draft_report
        -> verify --(fails, first time)--> draft_report
                 \\-> finalize

A fixed graph rather than a free-form agent: every run takes the same path, each node is
testable alone, and the trace records what each step saw and produced.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, TypedDict

from pydantic import ValidationError

from sentinel.common.config import CopilotParams
from sentinel.common.logging import get_logger
from sentinel.copilot import prompts
from sentinel.copilot.guard import Guard
from sentinel.copilot.models import Citation, Draft, IncidentReport, TechniqueRef, draft_schema
from sentinel.copilot.rag import Hit, Retriever
from sentinel.copilot.severity import assess
from sentinel.copilot.tools import AlertTools
from sentinel.llm.base import LLMError, LLMProvider
from sentinel.llm.redact import Redactor

log = get_logger(__name__)


class State(TypedDict, total=False):
    alert_id: int
    evidence_override: dict[str, Any] | None  # red-team: replace the alert's evidence
    alert: dict[str, Any]
    untrusted: dict[str, str]  # block id -> guarded text
    guard: list[dict[str, Any]]
    related: dict[str, Any]
    query: str
    techniques: list[Hit]
    cves: list[Hit]
    severity: dict[str, Any]
    draft: Draft | None
    raw_draft: Any
    errors: list[str]
    attempts: int
    usage: dict[str, Any]
    provider: str | None
    model: str | None
    trace: list[dict[str, Any]]
    report: IncidentReport


class AlertNotFoundError(LookupError):
    pass


@dataclass
class Copilot:
    tools: AlertTools
    retriever: Retriever
    llm: LLMProvider | None
    guard: Guard
    cfg: CopilotParams
    use_guard: bool = True  # False only for the red-team baseline
    use_hints: bool = True
    structural: bool = True  # delimiters + untrusted-data policy (False: red-team baseline)

    # -- nodes ------------------------------------------------------------------------------

    def guard_inputs(self, s: State) -> State:
        alert = self.tools.get_alert(s["alert_id"])
        if alert is not None and s.get("evidence_override") is not None:
            alert = {**alert, "evidence": s["evidence_override"]}
        if alert is None:
            raise AlertNotFoundError(s["alert_id"])
        fields: dict[str, str] = {}
        ev = alert.get("evidence") or {}
        if ev.get("email_text"):
            fields["email_body"] = ev["email_text"][:2500]
        if ev.get("urls"):
            fields["urls"] = "\n".join(u["url"] for u in ev["urls"][:20])
        if ev.get("log_lines"):
            fields["log_lines"] = "\n".join(ev["log_lines"][:20])
        if alert["source"] == "email" and alert.get("reasons"):
            # Sentences quoted from the email: untrusted, however they were selected.
            fields["suspicious_sentences"] = "\n".join(r["text"] for r in alert["reasons"])
            alert = {**alert, "reasons": None}
        untrusted, findings = {}, []
        for name, text in fields.items():
            if self.use_guard:
                res = self.guard.scan(text, name)
                untrusted[name] = res.sanitized
                findings.append(res.summary())
            else:
                untrusted[name] = text
        return {"alert": alert, "untrusted": untrusted, "guard": findings}

    def gather_context(self, s: State) -> State:
        related = self.tools.related_alerts(s["alert"], self.cfg.related_window_minutes)
        return {"related": related}

    def map_attack(self, s: State) -> State:
        query = prompts.retrieval_query(s["alert"], s["related"], self.use_hints)
        tech = self.retriever.search(query, kind="technique", k=self.cfg.n_techniques)
        cves = (
            self.retriever.search(query, kind="cve", k=self.cfg.n_cves) if self.cfg.n_cves else []
        )
        # Offer a CVE only on strong evidence; a weak candidate invites a made-up link.
        if self.retriever._rerank is not None:
            cves = [h for h in cves if h.score >= self.cfg.cve_min_score]
        return {"query": query, "techniques": tech, "cves": cves}

    def assess_severity(self, s: State) -> State:
        flagged = any(g["flagged_segments"] for g in s.get("guard", []))
        sev = assess(s["alert"], s["related"], self.cfg.assets, self.cfg.severity, flagged)
        return {"severity": sev}

    def draft_report(self, s: State) -> State:
        attempts = s.get("attempts", 0) + 1
        usage = dict(
            s.get("usage")
            or {
                "input_tokens": 0,
                "output_tokens": 0,
                "llm_seconds": 0.0,
                "calls": 0,
                "cached_calls": 0,
            }
        )
        if self.llm is None:
            return {
                "draft": None,
                "attempts": attempts,
                "errors": ["no LLM provider"],
                "provider": "template",
            }
        redactor = Redactor()
        context = redactor.redact(prompts.build_context(dict(s), self.structural))
        prompt = context
        if s.get("errors"):
            prompt += (
                "\n\nYOUR PREVIOUS ANSWER WAS REJECTED:\n- "
                + "\n- ".join(s["errors"])
                + "\nReturn a corrected report."
            )
        schema = draft_schema([h.id for h in s["techniques"]], [h.id for h in s["cves"]])
        try:
            system = prompts.SYSTEM if self.structural else prompts.SYSTEM_NAIVE
            resp = self.llm.generate(system, prompt, schema=schema, temperature=0.0)
            raw = redactor.restore(resp.json())
        except LLMError as exc:
            log.warning("draft failed: %s", exc)
            return {
                "draft": None,
                "attempts": attempts,
                "errors": [f"LLM error: {exc}"],
                "usage": usage,
            }
        usage["input_tokens"] += resp.input_tokens
        usage["output_tokens"] += resp.output_tokens
        usage["llm_seconds"] += resp.seconds
        usage["calls"] += 1
        usage["cached_calls"] += int(resp.cached)
        try:
            draft = Draft.model_validate(raw)
        except ValidationError as exc:
            errs = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()][:8]
            return {
                "draft": None,
                "raw_draft": raw,
                "attempts": attempts,
                "errors": errs,
                "usage": usage,
                "provider": resp.provider,
                "model": resp.model,
            }
        return {
            "draft": draft,
            "raw_draft": raw,
            "attempts": attempts,
            "errors": [],
            "usage": usage,
            "provider": resp.provider,
            "model": resp.model,
        }

    def verify(self, s: State) -> State:
        """Every cited ID must be in the retrieved or database context."""
        d = s.get("draft")
        if d is None:
            return {"errors": s.get("errors") or ["no draft"]}
        tech = {h.id for h in s["techniques"]}
        cves = {h.id for h in s["cves"]}
        alert_ids = {s["alert"]["id"]} | {r["id"] for r in s["related"].get("sample", [])}
        ips = {x for x in (s["alert"].get("src_ip"), s["alert"].get("dst_ip")) if x}
        for r in s["related"].get("sample", []):
            ips |= {x for x in (r.get("src_ip"), r.get("dst_ip")) if x}
        ips |= set(s["related"].get("top_sources", {})) | set(s["related"].get("top_targets", {}))
        errors = [
            f"technique {t.technique_id} is not a candidate"
            for t in d.techniques
            if t.technique_id not in tech
        ]
        errors += [f"CVE {c.cve_id} is not a candidate" for c in d.cves if c.cve_id not in cves]
        errors += [
            f"alert {i} is not in the context"
            for e in d.timeline
            for i in e.alert_ids
            if i not in alert_ids
        ]
        if s["alert"]["source"] == "network":
            errors += [
                f"host {h.ip} is not in the context" for h in d.affected_hosts if h.ip not in ips
            ]
        if not d.techniques and tech:
            errors.append("no technique chosen; pick the best-fitting candidate")
        return {"errors": errors}

    def route_after_verify(self, s: State) -> str:
        if s.get("errors") and s.get("attempts", 0) < 2 and self.llm is not None:
            return "draft_report"
        return "finalize"

    def finalize(self, s: State) -> State:
        d, errors = s.get("draft"), list(s.get("errors") or [])
        provider, model = s.get("provider"), s.get("model")
        dropped: list[str] = []
        if d is None:
            d, provider, model = self.template(s), "template", None
        else:
            # Keep the report, drop only what failed verification.
            tech = {h.id for h in s["techniques"]}
            cves = {h.id for h in s["cves"]}
            keep_t = [t for t in d.techniques if t.technique_id in tech]
            keep_c = [c for c in d.cves if c.cve_id in cves]
            dropped = [t.technique_id for t in d.techniques if t not in keep_t] + [
                c.cve_id for c in d.cves if c not in keep_c
            ]
            d = d.model_copy(update={"techniques": keep_t, "cves": keep_c})
        by_id = {h.id: h for h in [*s["techniques"], *s["cves"]]}
        cites = [
            Citation(kind="alert", id=str(s["alert"]["id"]), name=s["alert"]["predicted_family"])
        ]
        for ref in [*(t.technique_id for t in d.techniques), *(c.cve_id for c in d.cves)]:
            h = by_id[ref]
            cites.append(Citation(kind=h.kind, id=h.id, name=h.name, url=h.record.get("url")))
        report = IncidentReport(
            report_id=str(uuid.uuid4()),
            alert_id=s["alert"]["id"],
            draft=d,
            severity=s["severity"],
            citations=cites,
            guard=s.get("guard", []),
            verification={
                "passed": not errors,
                "errors": errors,
                "attempts": s.get("attempts", 0),
                "dropped_citations": dropped,
            },
            provider=provider,
            model=model,
            usage=s.get("usage") or {},
            candidates={
                "techniques": [h.id for h in s["techniques"]],
                "cves": [h.id for h in s["cves"]],
            },
        )
        return {"report": report}

    def template(self, s: State) -> Draft:
        """Deterministic report when no LLM is available: facts only, no prose model."""
        a, rel, sev = s["alert"], s["related"], s["severity"]
        top = s["techniques"][:1]
        where = (
            f"{a.get('src_ip')} -> {a.get('dst_ip')}:{a.get('dst_port')}"
            if a["source"] == "network"
            else f"email from {a.get('sender_domain') or 'unknown domain'}"
        )
        return Draft(
            title=f"{a['predicted_family']} alert {a['id']}",
            summary=(
                f"{a['predicted_family']} detected by the {a.get('detector')} detector "
                f"at {a['ts']} ({where}). {rel.get('count', 0)} related alerts in the window. "
                "Generated without an LLM: facts only."
            ),
            techniques=[
                TechniqueRef(technique_id=h.id, rationale="top retrieval match") for h in top
            ],
            severity_rationale="; ".join(f["detail"] for f in sev["factors"]),
            recommended_actions=["Review the alert and related activity manually."],
            limitations="No language model was available; mapping is the top retrieval hit.",
        )

    # -- graph ------------------------------------------------------------------------------

    def build(self) -> Any:
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(State)
        for name in (
            "guard_inputs",
            "gather_context",
            "map_attack",
            "assess_severity",
            "draft_report",
            "verify",
            "finalize",
        ):
            g.add_node(name, self._timed(name))
        g.add_edge(START, "guard_inputs")
        g.add_edge("guard_inputs", "gather_context")
        g.add_edge("gather_context", "map_attack")
        g.add_edge("map_attack", "assess_severity")
        g.add_edge("assess_severity", "draft_report")
        g.add_edge("draft_report", "verify")
        g.add_conditional_edges("verify", self.route_after_verify, ["draft_report", "finalize"])
        g.add_edge("finalize", END)
        return g.compile()

    def _timed(self, name: str) -> Any:
        fn = getattr(self, name)

        def node(s: State) -> State:
            t0 = time.perf_counter()
            out: State = fn(s)
            out["trace"] = [
                *s.get("trace", []),
                {"node": name, "seconds": round(time.perf_counter() - t0, 3)},
            ]
            return out

        return node

    def investigate(
        self, alert_id: int, evidence_override: dict[str, Any] | None = None
    ) -> tuple[IncidentReport, list[dict[str, Any]]]:
        start: State = {"alert_id": int(alert_id), "trace": []}
        if evidence_override is not None:
            start["evidence_override"] = evidence_override
        final = self.build().invoke(start)
        return final["report"], final["trace"]
