import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ReasonBars } from "../components/charts";
import { Button, Card, Empty, ErrorBox, Family, Loading, SeverityBadge } from "../components/ui";
import { api, fmtTime } from "../lib/api";

export default function AlertDetail() {
  const id = Number(useParams().id);
  const nav = useNavigate();
  const alert = useQuery({ queryKey: ["alert", id], queryFn: () => api.alert(id) });
  const related = useQuery({ queryKey: ["related", id], queryFn: () => api.related(id), enabled: alert.isSuccess });
  const investigate = useMutation({ mutationFn: () => api.investigate(id), onSuccess: (r) => nav(`/reports/${r.report_id}`) });

  if (alert.isLoading) return <Loading />;
  if (alert.error) return <ErrorBox error={alert.error} />;
  const a = alert.data!;
  const shared = (a.evidence as { explained_by?: number } | null)?.explained_by;
  const emailText = (a.evidence as { email_text?: string } | null)?.email_text;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">Alert #{a.id}: <Family name={a.predicted_family} /></h1>
          <div className="mt-1 flex items-center gap-2 text-sm text-ink-2"><SeverityBadge level={a.severity} /> {fmtTime(a.ts)} · detector: {a.detector}</div>
        </div>
        <Button onClick={() => investigate.mutate()} disabled={investigate.isPending}>
          {investigate.isPending ? "Investigating… (seconds with Gemini, ~1 min locally)" : "Investigate with Copilot"}
        </Button>
      </div>
      {investigate.error && <ErrorBox error={investigate.error} />}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Flow">
          <dl className="grid grid-cols-2 gap-y-1.5 text-sm">
            {a.source === "email" ? <><dt className="text-ink-2">Sender domain</dt><dd>{a.sender_domain}</dd></> : <>
              <dt className="text-ink-2">Source</dt><dd className="font-mono text-xs">{a.src_ip}:{a.src_port}</dd>
              <dt className="text-ink-2">Target</dt><dd className="font-mono text-xs">{a.dst_ip}:{a.dst_port}</dd>
              <dt className="text-ink-2">Protocol</dt><dd>{a.protocol}</dd></>}
            <dt className="text-ink-2">Confidence</dt><dd>{a.confidence == null ? "—" : a.confidence.toFixed(3)}</dd>
            <dt className="text-ink-2">Attack score</dt><dd>{a.attack_score?.toFixed(3) ?? "—"}</dd>
            <dt className="text-ink-2">Anomaly score</dt><dd>{a.anomaly_score?.toFixed(3) ?? "—"}</dd>
            <dt className="text-ink-2">Model</dt><dd className="truncate text-xs" title={a.model_version ?? ""}>{a.model_version}</dd>
          </dl>
          {a.features && (
            <details className="mt-3 text-sm"><summary className="cursor-pointer text-ink-2">Flow statistics</summary>
              <dl className="mt-2 grid grid-cols-2 gap-y-1 text-xs">
                {Object.entries(a.features).map(([k, v]) => (
                  <div key={k} className="contents"><dt className="text-ink-2">{k}</dt><dd>{v == null ? "—" : v.toLocaleString()}</dd></div>
                ))}
              </dl>
            </details>
          )}
        </Card>
        <Card title={a.source === "email" ? "Sentences that raised the score" : "Why the model flagged it (SHAP)"} className="lg:col-span-2">
          {a.reasons?.length ? <ReasonBars reasons={a.reasons} /> : <Empty>No explanation stored for this alert.</Empty>}
          {shared && <p className="mt-2 text-xs text-ink-2">Shared explanation: computed once for this campaign on <Link className="underline" to={`/alerts/${shared}`}>alert #{shared}</Link> (same source, target and family in the same micro-batch).</p>}
          {emailText && <details className="mt-3 text-sm"><summary className="cursor-pointer text-ink-2">Email text (untrusted)</summary><pre className="mt-2 whitespace-pre-wrap text-xs">{emailText}</pre></details>}
        </Card>
      </div>

      <Card title="Related activity (±30 min)">
        {related.isLoading ? <Loading /> : related.error ? <ErrorBox error={related.error} /> : (
          <>
            <p className="mb-2 text-sm text-ink-2">{related.data!.count.toLocaleString()} related alerts from the same source or to the same target.</p>
            <table className="w-full text-sm">
              <tbody>
                {related.data!.sample.map((r) => (
                  <tr key={r.id} className="border-t border-line">
                    <td className="py-1.5"><Link className="underline" to={`/alerts/${r.id}`}>#{r.id}</Link></td>
                    <td className="text-xs text-ink-2">{fmtTime(r.ts)}</td>
                    <td><Family name={r.predicted_family} /></td>
                    <td className="font-mono text-xs">{r.src_ip} → {r.dst_ip}:{r.dst_port}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </Card>
    </div>
  );
}
