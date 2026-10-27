import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button, Card, ErrorBox, SeverityBadge } from "../components/ui";
import { api, pct, type EmailFeedback } from "../lib/api";

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
    findings: Finding[];
  };
  intel: null | {
    domains: Array<{ domain: string; seen_as: string[]; registered: string | null; age_days: number | null; listed_on: string[] }>;
    findings: Finding[];
    feeds_loaded: boolean;
  };
  payment: Finding[];
  model_version: string | null;
  url_model_version: string | null;
};
type Finding = { check: string; severity: string; role: "alert" | "review" | "ignore"; text: string };

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

function Findings({ items }: { items: Finding[] }) {
  return (
    <table className="mt-2 w-full text-sm">
      <tbody>
        {items.map((f, i) => (
          <tr key={i} className="border-t border-line align-top">
            <td className="py-1.5 pr-3 break-all">{f.text}</td>
            <td className="whitespace-nowrap py-1.5 text-right text-xs text-ink-2">{f.severity} · {ROLE[f.role] ?? f.role}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function age(days: number | null, registered: string | null) {
  if (days == null) return registered ? `registered ${registered}` : "age unknown";
  return days < 365 ? `${days} days old` : `${Math.floor(days / 365)} years old`;
}

// "What is it really?": the analyst's label, saved with what was checked and what the models said.
function Verdict({ r, email }: { r: Result; email: EmailFeedback["email"] }) {
  const qc = useQueryClient();
  const send = useMutation({
    mutationFn: (verdict: EmailFeedback["verdict"]) =>
      api.emailFeedback({
        verdict,
        email,
        result: {
          verdict: r.verdict,
          probability: r.probability,
          flagged_by: r.flagged_by,
          checks: [...(r.sender?.findings ?? []), ...(r.intel?.findings ?? [])].map((f) => f.check),
          model_version: r.model_version,
          url_model_version: r.url_model_version,
        },
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["feedback-summary"] }),
  });
  return (
    <div className="mt-3 border-t border-line pt-3 text-sm">
      {send.data ? (
        <p className="text-ink-2">Saved. {send.data.model_was_right ? "The verdict was right." : "Recorded as a mistake: it counts in the live error rates on Model health."}</p>
      ) : (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-ink-2">What is it really?</span>
          <Button kind="plain" onClick={() => send.mutate("phishing")} disabled={send.isPending}>Phishing</Button>
          <Button kind="plain" onClick={() => send.mutate("legitimate")} disabled={send.isPending}>Legitimate</Button>
        </div>
      )}
      {send.error && <div className="mt-2"><ErrorBox error={send.error} /></div>}
    </div>
  );
}

export default function Phishing() {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [urls, setUrls] = useState("");
  const [sender, setSender] = useState("");
  const [replyTo, setReplyTo] = useState("");
  const [eml, setEml] = useState<{ name: string; text: string } | null>(null);
  // The email as scored, kept with the result so the analyst's verdict labels exactly that.
  const [checked, setChecked] = useState<EmailFeedback["email"] | null>(null);
  const m = useMutation({
    mutationFn: (email: EmailFeedback["email"]) => api.phishing(email) as Promise<Result>,
    onSuccess: (_, email) => setChecked(email),
  });
  const submit = () =>
    m.mutate({
      subject,
      body,
      urls: urls.split(/\s+/).filter(Boolean),
      sender,
      reply_to: replyTo,
      ...(eml ? { eml: eml.text } : {}),
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
        <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); submit(); }}>
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
            {/* Keyed by the result, so a new check shows fresh buttons. */}
            {checked && <Verdict key={m.submittedAt} r={r} email={checked} />}
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
          {r.payment.length > 0 && (
            <Card title="Payment request" className="lg:col-span-3">
              <p className="text-sm text-ink-2">
                The email asks for money. Invoice scams look like ordinary business mail, so before paying, confirm the request by phone, using a number you already have, not one from this email.
              </p>
              <Findings items={r.payment} />
            </Card>
          )}
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
                <Findings items={r.sender.findings} />
              ) : (
                <p className="mt-2 text-xs text-ink-2">Nothing unusual about the sender.</p>
              )}
            </Card>
          )}
          {r.intel && r.intel.domains.length > 0 && (
            <Card title="Domain reputation" className="lg:col-span-3">
              {!r.intel.feeds_loaded && <p className="mb-2 text-xs text-ink-2">Blocklists are still downloading; only domain age was checked.</p>}
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-ink-2"><tr><th className="py-1">Domain</th><th>Seen as</th><th>Age</th><th>Blocklists</th></tr></thead>
                <tbody>
                  {r.intel.domains.map((d) => (
                    <tr key={d.domain} className="border-t border-line">
                      <td className="py-1.5 font-mono text-xs break-all">{d.domain}</td>
                      <td className="text-ink-2">{d.seen_as.join(", ")}</td>
                      <td>{age(d.age_days, d.registered)}</td>
                      <td>{d.listed_on.length ? <span className="font-medium">✕ {d.listed_on.join(", ")}</span> : <span className="text-ink-2">not listed</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {r.intel.findings.length > 0 && <Findings items={r.intel.findings} />}
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
