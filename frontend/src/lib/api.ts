// Thin typed client for the Sentinel API (same origin: /app is served by the API).

export type Alert = {
  id: number;
  ts: string;
  source: "network" | "email";
  src_ip?: string | null;
  dst_ip?: string | null;
  src_port?: number | null;
  dst_port?: number | null;
  protocol?: number | null;
  sender_domain?: string | null;
  predicted_family: string;
  detector?: string | null;
  confidence?: number | null;
  attack_score?: number | null;
  anomaly_score?: number | null;
  severity?: string | null;
  reasons?: Reason[] | null;
  evidence?: Record<string, unknown> | null;
  features?: Record<string, number | null> | null;
  model_version?: string | null;
};
export type Reason = { feature?: string; value?: number; text: string; contribution: number };

export type Overview = {
  window_minutes: number;
  latest: string | null;
  total: number;
  per_minute: Array<{ minute: string } & Record<string, number | string>>;
  by_severity: Record<string, number>;
  by_family: Record<string, number>;
  top_sources: Array<{ src_ip: string; alerts: number }>;
};

export type Report = {
  report_id: string;
  alert_id: number;
  draft: {
    title: string;
    summary: string;
    timeline: Array<{ time: string; event: string; alert_ids: number[] }>;
    affected_hosts: Array<{ ip: string; role: string }>;
    techniques: Array<{ technique_id: string; rationale: string }>;
    cves: Array<{ cve_id: string; rationale: string }>;
    severity_rationale: string;
    recommended_actions: string[];
    injection_notes: string;
    limitations: string;
  };
  severity: { level: string; factors: Array<{ factor: string; effect: number; detail: string }> };
  citations: Array<{ kind: string; id: string; name: string; url?: string | null }>;
  guard: Array<{ field: string; flagged_segments: number; rules: string[] }>;
  verification: { passed: boolean; errors: string[]; attempts: number };
  provider: string | null;
  model: string | null;
  usage: { input_tokens?: number; output_tokens?: number; llm_seconds?: number };
  facts: Record<string, unknown>;
};

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const body = await r.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  overview: (minutes: number) => call<Overview>(`/stats/overview?minutes=${minutes}`),
  throughput: (minutes: number) =>
    call<{ available: boolean; points: Array<{ t: string; flows_per_s: number }> }>(`/stats/throughput?minutes=${minutes}`),
  alerts: (q: Record<string, string>) =>
    call<{ alerts: Alert[]; count: number }>(`/alerts?${new URLSearchParams(q)}`),
  alert: (id: number) => call<Alert>(`/alerts/${id}`),
  related: (id: number) => call<Record<string, unknown> & { count: number; sample: Alert[] }>(`/alerts/${id}/related`),
  investigate: (id: number) => call<Report>(`/copilot/investigate/${id}`, { method: "POST" }),
  report: (id: string) => call<Report>(`/reports/${id}`),
  phishing: (body: { subject: string; body: string; urls: string[] }) =>
    call<Record<string, unknown>>(`/score/phishing`, { method: "POST", body: JSON.stringify({ ...body, explain: true }) }),
  models: () => call<Record<string, unknown>>(`/models`),
  drift: () =>
    call<{
      threshold: number;
      trigger_window: string;
      sampled_flows: number;
      checks: Array<{ id: number; ts: string; window: string; n: number; share: number | null; drifted: Record<string, number> | null; triggered: number }>;
    }>(`/mlops/drift?limit=500`),
  retrains: () =>
    call<Array<{ id: number; started: string; finished: string | null; reason: string; n_extra: number | null; status: string; version: string | null; metrics: Record<string, unknown> | null; failed: boolean }>>(
      `/mlops/retrains`,
    ),
  study: () => call<Record<string, any>>(`/robustness/study`),
  robustness: (body: { pad: number; delay: number; family: string | null; n_samples: number }) =>
    call<Record<string, any>>(`/robustness/run`, { method: "POST", body: JSON.stringify(body) }),
};

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "medium" });
}
export const pct = (x: number | null | undefined, digits = 1) =>
  x == null ? "—" : `${(100 * x).toFixed(digits)}%`;
