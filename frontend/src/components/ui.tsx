import type { ReactNode } from "react";
import { SEVERITY_COLOR, familyColor, useDark } from "../lib/palette";

export function Card({ title, actions, children, className = "" }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-line bg-surface p-4 ${className}`}>
      {(title || actions) && (
        <header className="mb-3 flex items-center justify-between gap-2">
          {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

// A single headline number: a stat tile, not a chart.
export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-4 py-3">
      <div className="text-xs text-ink-2">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-ink">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-muted">{hint}</div>}
    </div>
  );
}

// Status colour always travels with its label (never colour alone).
export function SeverityBadge({ level }: { level?: string | null }) {
  if (!level) return <span className="text-xs text-muted">—</span>;
  const color = SEVERITY_COLOR[level] ?? "#898781";
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line px-2 py-0.5 text-xs font-medium text-ink">
      <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
      {level}
    </span>
  );
}

export function Family({ name }: { name: string }) {
  const dark = useDark();
  return (
    <span className="inline-flex items-center gap-1.5 text-ink">
      <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: familyColor(name, dark) }} />
      {name}
    </span>
  );
}

export function Loading({ what = "Loading" }: { what?: string }) {
  return <div className="py-6 text-center text-sm text-muted">{what}…</div>;
}

export function ErrorBox({ error }: { error: unknown }) {
  return (
    <div className="rounded-md border border-line p-3 text-sm text-ink" role="alert">
      <span className="font-semibold">Couldn't load this.</span> {error instanceof Error ? error.message : String(error)}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="py-6 text-center text-sm text-muted">{children}</div>;
}

export function Button({ children, onClick, disabled, kind = "primary", type = "button" }: { children: ReactNode; onClick?: () => void; disabled?: boolean; kind?: "primary" | "plain"; type?: "button" | "submit" }) {
  const base = "rounded-md px-3 py-1.5 text-sm font-medium transition disabled:opacity-50";
  const look = kind === "primary" ? "bg-ink text-surface hover:opacity-90" : "border border-line text-ink hover:bg-page";
  return (
    <button type={type} className={`${base} ${look}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}
