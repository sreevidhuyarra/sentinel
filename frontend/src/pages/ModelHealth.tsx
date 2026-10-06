import { useQuery } from "@tanstack/react-query";
import { DriftChart } from "../components/charts";
import { Card, Empty, ErrorBox, Loading, Stat } from "../components/ui";
import { api, fmtTime, pct } from "../lib/api";

const GRAFANA = `${location.protocol}//${location.hostname}:3000`;

// "models:/ids-classifier@production (v3)" -> "v3"; a local fallback directory -> "local".
function servedVersion(source: unknown): string {
  if (source == null) return "—";
  const v = /\(v(\d+)\)$/.exec(String(source));
  return v ? `v${v[1]}` : String(source).startsWith("models:/") ? "registry" : "local";
}

export default function ModelHealth() {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const drift = useQuery({ queryKey: ["drift"], queryFn: api.drift, refetchInterval: 15000 });
  const runs = useQuery({ queryKey: ["retrains"], queryFn: api.retrains, refetchInterval: 15000 });
  const last = drift.data?.checks.filter((c) => c.window === drift.data?.trigger_window && c.share != null).at(-1);
  const perf = drift.data?.checks.filter((c) => c.attack_recall != null || c.benign_fpr != null).at(-1);
  const m = models.data as Record<string, any> | undefined;

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Model health</h1>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Drift share (latest)" value={last ? pct(last.share) : "—"} hint={drift.data ? `retrain at ≥ ${pct(drift.data.threshold, 0)} (${drift.data.trigger_window} flows)` : undefined} />
        <Stat label="Drift window" value={last ? `${last.n.toLocaleString()} flows` : "—"} hint={drift.data ? `${drift.data.sampled_flows.toLocaleString()} sampled in total` : undefined} />
        <Stat label="Live attack recall" value={perf?.attack_recall != null ? pct(perf.attack_recall) : "—"} hint={drift.data ? `retrain below ${pct(drift.data.min_attack_recall, 0)} (labelled flows)` : undefined} />
        <Stat label="Live false alarms" value={perf?.benign_fpr != null ? pct(perf.benign_fpr) : "—"} hint={drift.data ? `retrain above ${pct(drift.data.max_benign_fpr, 0)}; ~1–2% by design` : undefined} />
        <Stat label="Retraining runs" value={runs.data?.length ?? "—"} hint={runs.data?.[0] ? `last: ${runs.data[0].status}` : undefined} />
        <Stat label="IDS model" value={servedVersion(m?.ids?.source)} hint={m?.ids?.source ? String(m.ids.source).replace(/ \(v\d+\)$/, "") : undefined} />
      </div>

      <Card title="Data drift of live flows vs the benign training reference">
        {drift.isLoading ? <Loading /> : drift.error ? <ErrorBox error={drift.error} /> :
          drift.data!.checks.length === 0 ? <Empty>No drift checks yet: run <code>sentinel mlops drift-watch</code> while traffic streams.</Empty> :
          <DriftChart checks={drift.data!.checks} threshold={drift.data!.threshold} />}
        {last?.drifted && Object.keys(last.drifted).length > 0 && (
          <p className="mt-2 text-xs text-ink-2">Most drifted now (PSI): {Object.entries(last.drifted).slice(0, 6).map(([f, v]) => `${f} ${v.toFixed(2)}`).join(" · ")}</p>
        )}
      </Card>

      <Card title="Retraining runs (Prefect flow)">
        {runs.isLoading ? <Loading /> : runs.error ? <ErrorBox error={runs.error} /> : runs.data!.length === 0 ? <Empty>No retraining yet.</Empty> : (
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-ink-2"><tr><th className="py-1">Run</th><th>Started</th><th>Status</th><th>Live flows added</th><th>Candidate macro-F1</th><th>Production macro-F1</th><th>Reason</th></tr></thead>
            <tbody>
              {runs.data!.map((r) => (
                <tr key={r.id} className="border-t border-line">
                  <td className="py-1.5">#{r.id}{r.version ? ` · v${r.version}` : ""}</td>
                  <td className="text-xs text-ink-2">{fmtTime(r.started)}</td>
                  <td>{r.status}</td>
                  <td>{r.n_extra?.toLocaleString() ?? "—"}</td>
                  <td>{r.metrics?.candidate_value != null ? Number(r.metrics.candidate_value).toFixed(4) : "—"}</td>
                  <td>{r.metrics?.production_value != null ? Number(r.metrics.production_value).toFixed(4) : "—"}</td>
                  <td className="text-xs text-ink-2">{r.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Card title="Serving models" actions={<a className="text-sm text-ink-2 underline" href={GRAFANA} target="_blank" rel="noreferrer">Open Grafana</a>}>
        {models.error ? <ErrorBox error={models.error} /> : (
          <dl className="grid gap-y-1 text-sm md:grid-cols-2">
            {["ids", "anomaly", "phishing", "url"].map((k) => (
              <div key={k} className="contents"><dt className="text-ink-2">{k}</dt><dd className="truncate">{m?.[k]?.source ?? "not loaded"}</dd></div>
            ))}
          </dl>
        )}
        <iframe title="Grafana overview" className="mt-4 h-[520px] w-full rounded-md border border-line"
                src={`${GRAFANA}/d/sentinel-overview/sentinel-overview?orgId=1&kiosk&refresh=10s`} />
      </Card>
    </div>
  );
}
