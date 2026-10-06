import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { Card, Empty, ErrorBox, Family, Loading, SeverityBadge } from "../components/ui";
import { api, fmtTime } from "../lib/api";
import { SEVERITIES } from "../lib/palette";

const FAMILIES = ["", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration", "Unknown anomaly", "Phishing/Spam"];

export default function Alerts() {
  const [family, setFamily] = useState("");
  const [severity, setSeverity] = useState("");
  const [src, setSrc] = useState("");
  const q: Record<string, string> = { limit: "100" };
  if (family) q.family = family;
  if (severity) q.severity = severity;
  if (src) q.src_ip = src;
  const res = useQuery({ queryKey: ["alerts", q], queryFn: () => api.alerts(q), refetchInterval: 10000 });
  const sel = "rounded-md border border-line bg-surface px-2 py-1 text-sm";
  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Alerts</h1>
      {/* Filters in one row above the table. */}
      <div className="flex flex-wrap gap-2">
        <select className={sel} value={family} onChange={(e) => setFamily(e.target.value)} aria-label="Family">
          {FAMILIES.map((f) => <option key={f} value={f}>{f || "All families"}</option>)}
        </select>
        <select className={sel} value={severity} onChange={(e) => setSeverity(e.target.value)} aria-label="Severity">
          <option value="">All severities</option>
          {SEVERITIES.map((s) => <option key={s}>{s}</option>)}
        </select>
        <input className={sel} placeholder="Source IP" value={src} onChange={(e) => setSrc(e.target.value.trim())} />
      </div>
      <Card>
        {res.isLoading ? <Loading /> : res.error ? <ErrorBox error={res.error} /> : res.data!.alerts.length === 0 ? <Empty>No alerts match.</Empty> : (
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-ink-2">
              <tr><th className="py-1">Alert</th><th>Time</th><th>Family</th><th>Severity</th><th>Source → target</th><th>Detector</th><th>Confidence</th></tr>
            </thead>
            <tbody>
              {res.data!.alerts.map((a) => (
                <tr key={a.id} className="border-t border-line">
                  <td className="py-1.5"><Link className="underline" to={`/alerts/${a.id}`}>#{a.id}</Link></td>
                  <td className="text-xs text-ink-2">{fmtTime(a.ts)}</td>
                  <td><Family name={a.predicted_family} /></td>
                  <td><SeverityBadge level={a.severity} /></td>
                  <td className="font-mono text-xs">{a.source === "email" ? a.sender_domain : `${a.src_ip} → ${a.dst_ip}:${a.dst_port}`}</td>
                  <td className="text-ink-2">{a.detector}</td>
                  <td>{a.confidence == null ? "—" : a.confidence.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
