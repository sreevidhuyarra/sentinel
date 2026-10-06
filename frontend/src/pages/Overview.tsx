import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { AlertsPerMinute, Throughput } from "../components/charts";
import { Card, Empty, ErrorBox, Family, Loading, SeverityBadge, Stat } from "../components/ui";
import { api } from "../lib/api";
import { useLiveAlerts } from "../lib/live";
import { SEVERITIES, sortFamilies } from "../lib/palette";

const WINDOWS = [15, 60, 240];

export default function Overview() {
  const [minutes, setMinutes] = useState(60);
  const ov = useQuery({ queryKey: ["overview", minutes], queryFn: () => api.overview(minutes), refetchInterval: 5000 });
  const tp = useQuery({ queryKey: ["throughput"], queryFn: () => api.throughput(30), refetchInterval: 5000 });
  const live = useLiveAlerts(40);

  const families = sortFamilies(Object.keys(ov.data?.by_family ?? {}));
  const rate = tp.data?.points.at(-1)?.flows_per_s;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-lg font-semibold">Live overview</h1>
        <div className="flex items-center gap-1 text-sm" role="group" aria-label="Time window">
          {WINDOWS.map((w) => (
            <button key={w} onClick={() => setMinutes(w)}
                    className={`rounded-md px-2.5 py-1 ${w === minutes ? "bg-ink text-surface" : "text-ink-2 hover:bg-surface"}`}>
              {w < 60 ? `${w} min` : `${w / 60} h`}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Flows scored / s" value={rate == null ? "—" : rate.toFixed(0)}
              hint={tp.data && !tp.data.available ? "Prometheus not reachable" : "last minute"} />
        <Stat label={`Alerts (last ${minutes} min of traffic)`} value={ov.data?.total.toLocaleString() ?? "—"} />
        <Stat label="Most frequent family" value={families[0] ? <Family name={Object.entries(ov.data!.by_family)[0][0]} /> : "—"} />
        <Stat label="Live feed" value={live.connected ? "Connected" : "Reconnecting"} hint="WebSocket /ws/alerts" />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Alerts per minute by family" className="lg:col-span-2">
          {ov.isLoading ? <Loading /> : ov.error ? <ErrorBox error={ov.error} /> :
            ov.data!.total === 0 ? <Empty>No alerts yet. Start the replayer and the stream detector.</Empty> :
            <AlertsPerMinute data={ov.data!.per_minute} families={families} />}
        </Card>
        <Card title="By family">
          {/* Table view: the text identity for every colour (relief for low-contrast hues). */}
          <table className="w-full text-sm">
            <tbody>
              {families.map((f) => (
                <tr key={f} className="border-b border-line last:border-0">
                  <td className="py-1.5"><Family name={f} /></td>
                  <td className="py-1.5 text-right">{ov.data!.by_family[f].toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3 className="mt-4 mb-1 text-xs font-semibold text-ink-2">By severity</h3>
          <div className="flex flex-wrap gap-2">
            {SEVERITIES.filter((s) => ov.data?.by_severity[s]).map((s) => (
              <span key={s} className="text-sm"><SeverityBadge level={s} /> {ov.data!.by_severity[s].toLocaleString()}</span>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Flows scored per second (last 30 min)" className="lg:col-span-2">
          {tp.data && tp.data.points.length ? <Throughput points={tp.data.points} /> :
            <Empty>{tp.data && !tp.data.available ? "Prometheus is not running." : "No traffic yet."}</Empty>}
        </Card>
        <Card title="Top sources">
          {ov.data?.top_sources.length ? (
            <table className="w-full text-sm">
              <tbody>
                {ov.data.top_sources.map((t) => (
                  <tr key={t.src_ip} className="border-b border-line last:border-0">
                    <td className="py-1.5 font-mono text-xs">{t.src_ip}</td>
                    <td className="py-1.5 text-right">{t.alerts.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <Empty>—</Empty>}
        </Card>
      </div>

      <Card title="Live alert feed" actions={<Link to="/alerts" className="text-sm text-ink-2 underline">All alerts</Link>}>
        {live.alerts.length === 0 ? <Empty>Waiting for new alerts…</Empty> : (
          <div className="max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-ink-2">
                <tr><th className="py-1">Alert</th><th>Family</th><th>Severity</th><th>Source → target</th><th>Confidence</th></tr>
              </thead>
              <tbody>
                {live.alerts.map((a) => (
                  <tr key={a.id} className="border-t border-line">
                    <td className="py-1.5"><Link className="underline" to={`/alerts/${a.id}`}>#{a.id}</Link></td>
                    <td><Family name={a.predicted_family} /></td>
                    <td><SeverityBadge level={a.severity} /></td>
                    <td className="font-mono text-xs">{a.source === "email" ? a.sender_domain : `${a.src_ip} → ${a.dst_ip}:${a.dst_port}`}</td>
                    <td>{a.confidence == null ? "—" : a.confidence.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
