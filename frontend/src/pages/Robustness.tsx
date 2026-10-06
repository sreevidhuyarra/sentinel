import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { EvasionBars } from "../components/charts";
import { Button, Card, ErrorBox, Family, Loading } from "../components/ui";
import { api, pct } from "../lib/api";

const NAMES: Record<string, string> = {
  ensemble: "Deployed ensemble",
  ensemble_plus_review: "Ensemble + review flag",
  lightgbm: "LightGBM",
  mlp: "MLP",
  mlp_adv_trained: "MLP, adversarially trained",
  lightgbm_no_timing: "LightGBM, no timing features",
  lightgbm_no_controllable: "LightGBM, no controllable features",
};
const FAMILIES = ["", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration"];

export default function Robustness() {
  const study = useQuery({ queryKey: ["study"], queryFn: api.study });
  const [budget, setBudget] = useState("high");
  const [pad, setPad] = useState(0.5);
  const [delay, setDelay] = useState(3);
  const [family, setFamily] = useState("");
  const run = useMutation({ mutationFn: () => api.robustness({ pad, delay, family: family || null, n_samples: 300 }) });
  const s = study.data;
  const rows = s ? Object.entries(s.problem_space[budget] as Record<string, number | null>)
    .filter(([, v]) => v != null).map(([k, v]) => ({ name: NAMES[k] ?? k, value: v as number }))
    .sort((a, b) => b.value - a.value) : [];
  const sel = "rounded-md border border-line bg-surface px-2 py-1 text-sm";

  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Robustness lab</h1>
      <Card title="Module 4 study: realistic padding + delay attacks (share of caught attacks that evade)"
            actions={
              <div className="flex gap-1" role="group" aria-label="Attacker budget">
                {["low", "medium", "high"].map((b) => (
                  <button key={b} onClick={() => setBudget(b)} className={`rounded-md px-2.5 py-1 text-sm ${b === budget ? "bg-ink text-surface" : "text-ink-2"}`}>{b}</button>
                ))}
              </div>}>
        {study.isLoading ? <Loading /> : study.error ? <ErrorBox error={study.error} /> : <EvasionBars rows={rows} />}
        <p className="mt-2 text-xs text-ink-2">Budgets: low ≤ 10% padding and 1.5× slower; medium ≤ 50% and 3×; high ≤ 100% and 10×. {s ? `${s.targets.total.toLocaleString()} test attack flows.` : ""}</p>
      </Card>

      <Card title="Run it live against the deployed detector">
        <form className="flex flex-wrap items-end gap-4 text-sm" onSubmit={(e) => { e.preventDefault(); run.mutate(); }}>
          <label className="flex flex-col gap-1">Padding (extra payload) <span className="text-ink-2">{Math.round(pad * 100)}%</span>
            <input type="range" min={0} max={2} step={0.05} value={pad} onChange={(e) => setPad(Number(e.target.value))} /></label>
          <label className="flex flex-col gap-1">Slow-down factor <span className="text-ink-2">{delay.toFixed(1)}×</span>
            <input type="range" min={1} max={20} step={0.5} value={delay} onChange={(e) => setDelay(Number(e.target.value))} /></label>
          <label className="flex flex-col gap-1">Family
            <select className={sel} value={family} onChange={(e) => setFamily(e.target.value)}>
              {FAMILIES.map((f) => <option key={f} value={f}>{f || "All families"}</option>)}
            </select></label>
          <Button type="submit" disabled={run.isPending}>{run.isPending ? "Attacking 300 flows…" : "Run attack"}</Button>
        </form>
        {run.error && <div className="mt-3"><ErrorBox error={run.error} /></div>}
        {run.data && (
          <div className="mt-4">
            <div className="text-2xl font-semibold">{pct(run.data.evasion_rate)} evaded</div>
            <div className="text-sm text-ink-2">{run.data.evaded_after} of {run.data.caught_before} attacks the detector caught now pass as benign. {run.data.note}</div>
            <table className="mt-3 w-full text-sm">
              <tbody>
                {Object.entries(run.data.by_family as Record<string, { evaded: number; caught_before: number; rate: number }>).map(([f, v]) => (
                  <tr key={f} className="border-t border-line"><td className="py-1.5"><Family name={f} /></td><td className="text-right">{v.evaded} / {v.caught_before}</td><td className="text-right">{pct(v.rate)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
