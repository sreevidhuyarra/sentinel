import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Reason } from "../lib/api";
import { chartInk, familyColor, sortFamilies, useDark } from "../lib/palette";

// Chart conventions (dataviz): one measure per chart (no dual axes), thin marks, 2px lines,
// recessive grid, a legend for >= 2 series, tooltips on hover, family colours fixed by entity.

function axisProps(dark: boolean) {
  const c = chartInk(dark);
  return { stroke: c.grid, tick: { fill: c.muted, fontSize: 11 }, tickLine: false };
}

function tooltipStyle(dark: boolean) {
  const c = chartInk(dark);
  return {
    contentStyle: { background: c.surface, border: `1px solid ${c.grid}`, borderRadius: 6, color: c.ink, fontSize: 12 },
    labelStyle: { color: c.ink2 },
    cursor: { fill: dark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)" },
  };
}

const num = (v: unknown): number => Number(Array.isArray(v) ? v[0] : (v ?? 0));

const minuteLabel = (iso: string) => new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

export function AlertsPerMinute({ data, families }: { data: Array<Record<string, any>>; families: string[] }) {
  const dark = useDark();
  const c = chartInk(dark);
  const order = sortFamilies(families);
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barCategoryGap={2}>
        <CartesianGrid vertical={false} stroke={c.grid} />
        <XAxis dataKey="minute" tickFormatter={minuteLabel} {...axisProps(dark)} minTickGap={24} />
        <YAxis allowDecimals={false} {...axisProps(dark)} width={44} />
        <Tooltip {...tooltipStyle(dark)} labelFormatter={(l) => minuteLabel(String(l))} />
        <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => <span style={{ color: c.ink2 }}>{v}</span>} />
        {order.map((f, i) => (
          <Bar key={f} dataKey={f} stackId="a" fill={familyColor(f, dark)} stroke={c.surface} strokeWidth={1}
               radius={i === order.length - 1 ? [4, 4, 0, 0] : 0} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Throughput({ points }: { points: Array<{ t: string; flows_per_s: number }> }) {
  const dark = useDark();
  const c = chartInk(dark);
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid vertical={false} stroke={c.grid} />
        <XAxis dataKey="t" tickFormatter={minuteLabel} {...axisProps(dark)} minTickGap={32} />
        <YAxis {...axisProps(dark)} width={44} />
        <Tooltip {...tooltipStyle(dark)} labelFormatter={(l) => minuteLabel(String(l))}
                 formatter={(v) => [`${num(v).toFixed(0)} flows/s`, "scored"]} />
        <Line type="monotone" dataKey="flows_per_s" stroke={familyColor("DoS", dark)} strokeWidth={2} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// SHAP / anomaly reasons: signed contributions as horizontal bars, labelled directly.
export function ReasonBars({ reasons }: { reasons: Reason[] }) {
  const dark = useDark();
  const c = chartInk(dark);
  const data = reasons.map((r) => ({ label: r.text, value: r.contribution }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(120, 36 * data.length + 30)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
        <CartesianGrid horizontal={false} stroke={c.grid} />
        <XAxis type="number" {...axisProps(dark)} />
        <YAxis type="category" dataKey="label" width={260} {...axisProps(dark)} tick={{ fill: c.ink2, fontSize: 12 }} />
        <ReferenceLine x={0} stroke={c.muted} />
        <Tooltip {...tooltipStyle(dark)} formatter={(v) => [num(v).toFixed(3), "contribution"]} />
        <Bar dataKey="value" isAnimationActive={false} radius={4}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.value >= 0 ? familyColor("DoS", dark) : c.muted} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DriftChart({ checks, threshold }: { checks: Array<{ ts: string; window: string; share: number | null }>; threshold: number }) {
  const dark = useDark();
  const c = chartInk(dark);
  const byTs = new Map<string, Record<string, any>>();
  for (const ch of checks) {
    if (ch.share == null) continue;
    const row = byTs.get(ch.ts) ?? { ts: ch.ts };
    row[ch.window] = ch.share;
    byTs.set(ch.ts, row);
  }
  const data = [...byTs.values()];
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid vertical={false} stroke={c.grid} />
        <XAxis dataKey="ts" tickFormatter={minuteLabel} {...axisProps(dark)} minTickGap={32} />
        <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axisProps(dark)} width={44} />
        <Tooltip {...tooltipStyle(dark)} labelFormatter={(l) => minuteLabel(String(l))} formatter={(v, n) => [`${(num(v) * 100).toFixed(1)}%`, `${n} flows`]} />
        <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => <span style={{ color: c.ink2 }}>{v === "all" ? "all flows" : "flows passed as benign"}</span>} />
        <ReferenceLine y={threshold} stroke={c.ink2} strokeDasharray="4 4" label={{ value: `retrain threshold ${Math.round(threshold * 100)}%`, fill: c.ink2, fontSize: 11, position: "insideTopRight" }} />
        <Line type="monotone" dataKey="all" stroke={familyColor("DoS", dark)} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
        <Line type="monotone" dataKey="benign" stroke={familyColor("DDoS", dark)} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// One measure (evasion rate) per bar, one bar per configuration.
export function EvasionBars({ rows }: { rows: Array<{ name: string; value: number }> }) {
  const dark = useDark();
  const c = chartInk(dark);
  return (
    <ResponsiveContainer width="100%" height={Math.max(160, 34 * rows.length + 30)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 40, bottom: 4, left: 8 }}>
        <CartesianGrid horizontal={false} stroke={c.grid} />
        <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} {...axisProps(dark)} />
        <YAxis type="category" dataKey="name" width={200} {...axisProps(dark)} tick={{ fill: c.ink2, fontSize: 12 }} />
        <Tooltip {...tooltipStyle(dark)} formatter={(v) => [`${(num(v) * 100).toFixed(1)}%`, "evaded"]} />
        <Bar dataKey="value" fill={familyColor("DoS", dark)} radius={4} isAnimationActive={false}
             label={{ position: "right", fill: c.ink2, fontSize: 11, formatter: (v: unknown) => `${(num(v) * 100).toFixed(1)}%` }} />
      </BarChart>
    </ResponsiveContainer>
  );
}
