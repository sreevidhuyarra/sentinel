import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { Button, Card, ErrorBox, Loading, SeverityBadge } from "../components/ui";
import type { Report as ReportT } from "../lib/api";
import { api } from "../lib/api";

function toMarkdown(r: ReportT): string {
  const d = r.draft;
  const cite = (id: string) => r.citations.find((c) => c.id === id);
  const lines = [
    `# ${d.title}`,
    "",
    `Severity: **${r.severity.level}** · alert #${r.alert_id} · ${r.provider ?? "?"} ${r.model ?? ""}`,
    "",
    d.summary,
    "",
    "## Facts",
    ...Object.entries(r.facts).map(([k, v]) => `- ${k}: ${Array.isArray(v) ? v.join("; ") : typeof v === "object" ? JSON.stringify(v) : v}`),
    "",
    "## Timeline",
    ...d.timeline.map((e) => `- ${e.time}: ${e.event}${e.alert_ids.length ? ` (alerts ${e.alert_ids.join(", ")})` : ""}`),
    "",
    "## ATT&CK techniques",
    ...d.techniques.map((t) => `- [${t.technique_id} ${cite(t.technique_id)?.name ?? ""}](${cite(t.technique_id)?.url ?? ""}): ${t.rationale}`),
    ...(d.cves.length ? ["", "## CVEs", ...d.cves.map((c) => `- [${c.cve_id}](${cite(c.cve_id)?.url ?? ""}): ${c.rationale}`)] : []),
    "",
    "## Severity",
    d.severity_rationale,
    ...r.severity.factors.map((f) => `- ${f.factor} (${f.effect >= 0 ? "+" : ""}${f.effect}): ${f.detail}`),
    "",
    "## Recommended actions",
    ...d.recommended_actions.map((a, i) => `${i + 1}. ${a}`),
    ...(d.injection_notes ? ["", "## Prompt-injection notes", d.injection_notes] : []),
    ...(d.limitations ? ["", "## Limitations", d.limitations] : []),
  ];
  return lines.join("\n") + "\n";
}

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

export default function Report() {
  const id = useParams().id!;
  const q = useQuery({ queryKey: ["report", id], queryFn: () => api.report(id) });
  if (q.isLoading) return <Loading what="Loading report" />;
  if (q.error) return <ErrorBox error={q.error} />;
  const r = q.data!;
  const d = r.draft;
  const cite = (cid: string) => r.citations.find((c) => c.id === cid);
  const flagged = r.guard.filter((g) => g.flagged_segments > 0);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">{d.title}</h1>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-ink-2">
            <SeverityBadge level={r.severity.level} />
            <Link className="underline" to={`/alerts/${r.alert_id}`}>alert #{r.alert_id}</Link>
            <span>· written by {r.provider}{r.model ? ` (${r.model})` : ""}</span>
            <span>· {r.verification.passed ? "verified: every citation is in the context" : `verification issues: ${r.verification.errors.join("; ")}`}</span>
          </div>
        </div>
        <div className="flex gap-2">
          <Button kind="plain" onClick={() => download(`incident-${r.alert_id}.md`, toMarkdown(r))}>Export Markdown</Button>
          <Button kind="plain" onClick={() => window.print()}>Print / PDF</Button>
        </div>
      </div>

      {flagged.length > 0 && (
        <div className="rounded-md border border-line p-3 text-sm" role="note">
          <span className="font-semibold">Prompt injection removed.</span> The guard redacted suspicious instructions in: {flagged.map((g) => g.field).join(", ")}. {d.injection_notes}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Summary" className="lg:col-span-2"><p className="text-sm leading-6">{d.summary}</p></Card>
        <Card title="Facts (from the database)">
          <dl className="grid grid-cols-2 gap-y-1 text-xs">
            {Object.entries(r.facts).filter(([, v]) => v != null && v !== "" && !(Array.isArray(v) && !v.length)).map(([k, v]) => (
              <div key={k} className="contents"><dt className="text-ink-2">{k.replaceAll("_", " ")}</dt><dd className="break-words">{Array.isArray(v) ? v.join("; ") : typeof v === "object" ? JSON.stringify(v) : String(v)}</dd></div>
            ))}
          </dl>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="ATT&CK techniques">
          <ul className="space-y-2 text-sm">
            {d.techniques.map((t) => (
              <li key={t.technique_id}>
                <a className="font-medium underline" href={cite(t.technique_id)?.url ?? "#"} target="_blank" rel="noreferrer">{t.technique_id} {cite(t.technique_id)?.name}</a>
                <div className="text-ink-2">{t.rationale}</div>
              </li>
            ))}
            {d.cves.map((c) => (
              <li key={c.cve_id}>
                <a className="font-medium underline" href={cite(c.cve_id)?.url ?? "#"} target="_blank" rel="noreferrer">{c.cve_id}</a>
                <div className="text-ink-2">{c.rationale}</div>
              </li>
            ))}
          </ul>
        </Card>
        <Card title={<span>Severity: {r.severity.level} (rule-based)</span>}>
          <p className="mb-2 text-sm">{d.severity_rationale}</p>
          <ul className="text-sm text-ink-2">
            {r.severity.factors.map((f, i) => <li key={i}>{f.effect > 0 ? "▲" : f.effect < 0 ? "▼" : "•"} {f.factor}: {f.detail}</li>)}
          </ul>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Timeline">
          <ol className="space-y-1.5 text-sm">
            {d.timeline.map((e, i) => (
              <li key={i}><span className="text-ink-2">{e.time}</span> {e.event} {e.alert_ids.map((aid) => <Link key={aid} className="underline" to={`/alerts/${aid}`}>#{aid}</Link>)}</li>
            ))}
          </ol>
        </Card>
        <Card title="Recommended actions">
          <ol className="list-decimal space-y-1 pl-5 text-sm">{d.recommended_actions.map((a, i) => <li key={i}>{a}</li>)}</ol>
          {d.limitations && <p className="mt-3 text-xs text-ink-2">Limitations: {d.limitations}</p>}
        </Card>
      </div>
      <p className="text-xs text-muted">{r.usage.input_tokens ?? 0} tokens in · {r.usage.output_tokens ?? 0} out · {(r.usage.llm_seconds ?? 0).toFixed(1)} s LLM time · prompts are redacted before they leave the machine.</p>
    </div>
  );
}
