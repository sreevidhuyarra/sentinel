import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { Button, Card, ErrorBox, SeverityBadge } from "../components/ui";
import { api, pct } from "../lib/api";

type Result = {
  is_malicious: boolean;
  flagged_by: string[];
  probability: number;
  threshold: number;
  severity: string | null;
  reasons: Array<{ text: string; contribution: number }>;
  urls: string[];
  url_results: Array<{ url: string; is_malicious: boolean; probability: number; reasons: Array<{ text: string }> }>;
};

function highlight(body: string, reasons: Result["reasons"]) {
  // Mark the sentences whose removal most lowered the score.
  let parts: Array<{ text: string; hit: boolean }> = [{ text: body, hit: false }];
  for (const r of reasons) {
    parts = parts.flatMap((p) => {
      if (p.hit || !r.text || !p.text.includes(r.text)) return [p];
      const [a, ...rest] = p.text.split(r.text);
      return [{ text: a, hit: false }, { text: r.text, hit: true }, { text: rest.join(r.text), hit: false }];
    });
  }
  return parts.map((p, i) => (p.hit ? <mark key={i} className="rounded bg-[#fab219]/40 px-0.5 text-ink">{p.text}</mark> : <span key={i}>{p.text}</span>));
}

export default function Phishing() {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [urls, setUrls] = useState("");
  const m = useMutation({
    mutationFn: () => api.phishing({ subject, body, urls: urls.split(/\s+/).filter(Boolean) }) as Promise<Result>,
  });
  const r = m.data;
  const input = "w-full rounded-md border border-line bg-surface px-3 py-2 text-sm";
  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Phishing check</h1>
      <Card>
        <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); m.mutate(); }}>
          <input className={input} placeholder="Subject" value={subject} onChange={(e) => setSubject(e.target.value)} />
          <textarea className={`${input} h-48 font-mono`} placeholder="Paste the email body" value={body} onChange={(e) => setBody(e.target.value)} required />
          <input className={input} placeholder="Extra URLs to check (space separated, optional)" value={urls} onChange={(e) => setUrls(e.target.value)} />
          <Button type="submit" disabled={!body || m.isPending}>{m.isPending ? "Scoring…" : "Check"}</Button>
        </form>
      </Card>
      {m.error && <ErrorBox error={m.error} />}
      {r && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card title="Verdict">
            <div className="text-2xl font-semibold">{r.is_malicious ? "Malicious" : "Looks legitimate"}</div>
            <div className="mt-1 text-sm text-ink-2">P(malicious) {pct(r.probability, 2)} · threshold {pct(r.threshold, 1)}</div>
            <div className="mt-2 flex items-center gap-2 text-sm"><SeverityBadge level={r.severity} /> {r.flagged_by.length ? `flagged by ${r.flagged_by.join(" + ")}` : ""}</div>
          </Card>
          <Card title="Evidence in the text" className="lg:col-span-2">
            <div className="whitespace-pre-wrap text-sm leading-6">{highlight(body, r.reasons)}</div>
            {!r.reasons.length && <p className="text-xs text-ink-2">No single sentence stands out.</p>}
          </Card>
          {r.url_results.length > 0 && (
            <Card title="Links" className="lg:col-span-3">
              <table className="w-full text-sm">
                <tbody>
                  {r.url_results.map((u) => (
                    <tr key={u.url} className="border-t border-line">
                      <td className="py-1.5 break-all font-mono text-xs">{u.url}</td>
                      <td className="text-right">{u.is_malicious ? "malicious" : "ok"} · {pct(u.probability, 1)}</td>
                      <td className="pl-3 text-xs text-ink-2">{u.reasons.map((x) => x.text).join("; ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
