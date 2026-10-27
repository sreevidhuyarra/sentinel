import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { Button, Card, ErrorBox, SeverityBadge } from "../components/ui";
import { api, pct } from "../lib/api";

type Result = {
  is_malicious: boolean;
  flagged_by: string[];
  verdict: "malicious" | "suspicious" | "clean";
  suspicious_because: string[];
  probability: number;
  threshold: number;
  suspicious_threshold: number;
  severity: string | null;
  reasons: Array<{ text: string; contribution: number }>;
  urls: string[];
  url_results: Array<{ url: string; is_malicious: boolean; probability: number; reasons: Array<{ text: string }> }>;
  subject: string;
  sender: null | {
    from_line: string;
    address: string;
    domain: string;
    reply_to: string;
    auth: Record<string, string>;
    findings: Array<{ check: string; severity: string; role: "alert" | "review" | "ignore"; text: string }>;
  };
};

const ROLE: Record<string, string> = { alert: "raises an alert", review: "marks for review", ignore: "not counted" };

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

const VERDICT: Record<Result["verdict"], string> = {
  malicious: "Malicious",
  suspicious: "Suspicious: review",
  clean: "Looks legitimate",
};

export default function Phishing() {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [urls, setUrls] = useState("");
  const [sender, setSender] = useState("");
  const [replyTo, setReplyTo] = useState("");
  const [eml, setEml] = useState<{ name: string; text: string } | null>(null);
  const m = useMutation({
    mutationFn: () =>
      api.phishing({
        subject,
        body,
        urls: urls.split(/\s+/).filter(Boolean),
        sender,
        reply_to: replyTo,
        ...(eml ? { eml: eml.text } : {}),
      }) as Promise<Result>,
  });
  const loadEml = async (file: File | undefined) => {
    if (!file) return;
    setEml({ name: file.name, text: await file.text() });
  };
  const r = m.data;
  const input = "w-full rounded-md border border-line bg-surface px-3 py-2 text-sm";
  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Phishing check</h1>
      <Card>
        <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); m.mutate(); }}>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <label className="cursor-pointer rounded-md border border-line px-3 py-1.5 font-medium hover:bg-page">
              Upload .eml
              <input type="file" accept=".eml,message/rfc822" className="hidden" onChange={(e) => loadEml(e.target.files?.[0])} />
            </label>
            {eml ? (
              <span className="text-ink-2">
                {eml.name}: subject, body, sender and SPF/DKIM/DMARC results are read from the file{" "}
                <button type="button" className="underline" onClick={() => setEml(null)}>remove</button>
              </span>
            ) : (
              <span className="text-ink-2">or paste the email below. An .eml includes the headers that show who really sent it.</span>
            )}
          </div>
          {!eml && (
            <>
              <div className="grid gap-3 sm:grid-cols-2">
                <input className={input} placeholder='From (optional), e.g. PayPal <service@paypal.com>' value={sender} onChange={(e) => setSender(e.target.value)} />
                <input className={input} placeholder="Reply-To (optional)" value={replyTo} onChange={(e) => setReplyTo(e.target.value)} />
              </div>
              <input className={input} placeholder="Subject" value={subject} onChange={(e) => setSubject(e.target.value)} />
              <textarea className={`${input} h-48 font-mono`} placeholder="Paste the email body" value={body} onChange={(e) => setBody(e.target.value)} />
            </>
          )}
          <input className={input} placeholder="Extra URLs to check (space separated, optional)" value={urls} onChange={(e) => setUrls(e.target.value)} />
          <Button type="submit" disabled={(!body && !eml) || m.isPending}>{m.isPending ? "Scoring…" : "Check"}</Button>
        </form>
      </Card>
      {m.error && <ErrorBox error={m.error} />}
      {r && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card title="Verdict">
            <div className="text-2xl font-semibold">{VERDICT[r.verdict]}</div>
            <div className="mt-1 text-sm text-ink-2">
              P(malicious) {pct(r.probability, 2)} · alert at {pct(r.threshold, 1)} · review from {pct(r.suspicious_threshold, 0)}
            </div>
            <div className="mt-2 flex items-center gap-2 text-sm"><SeverityBadge level={r.severity} /> {r.flagged_by.length ? `flagged by ${r.flagged_by.join(" + ")}` : ""}</div>
            {r.suspicious_because.length > 0 && (
              <ul className="mt-2 list-disc pl-5 text-xs text-ink-2">
                {r.suspicious_because.map((b) => <li key={b} className="break-all">{b}</li>)}
              </ul>
            )}
          </Card>
          <Card title="Evidence in the text" className="lg:col-span-2">
            {eml ? (
              <>
                <div className="mb-2 text-sm font-medium">{r.subject}</div>
                <ul className="list-disc space-y-1 pl-5 text-sm">
                  {r.reasons.map((x) => <li key={x.text}><mark className="rounded bg-[#fab219]/40 px-0.5 text-ink">{x.text}</mark></li>)}
                </ul>
              </>
            ) : (
              <div className="whitespace-pre-wrap text-sm leading-6">{highlight(body, r.reasons)}</div>
            )}
            {!r.reasons.length && <p className="text-xs text-ink-2">No single sentence stands out.</p>}
          </Card>
          {r.sender && (
            <Card title="Sender" className="lg:col-span-3">
              <div className="text-sm">
                <span className="font-mono text-xs break-all">{r.sender.from_line || "(no From line)"}</span>
                {r.sender.reply_to && <span className="ml-3 text-xs text-ink-2">Reply-To <span className="font-mono">{r.sender.reply_to}</span></span>}
              </div>
              {Object.keys(r.sender.auth).length > 0 && (
                <div className="mt-1 text-xs text-ink-2">
                  {Object.entries(r.sender.auth).map(([k, v]) => `${k.toUpperCase()} ${v}`).join(" · ")}
                </div>
              )}
              {r.sender.findings.length ? (
                <table className="mt-2 w-full text-sm">
                  <tbody>
                    {r.sender.findings.map((f) => (
                      <tr key={f.check} className="border-t border-line align-top">
                        <td className="py-1.5 pr-3">{f.text}</td>
                        <td className="whitespace-nowrap py-1.5 text-right text-xs text-ink-2">{f.severity} · {ROLE[f.role] ?? f.role}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="mt-2 text-xs text-ink-2">Nothing unusual about the sender.</p>
              )}
            </Card>
          )}
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
