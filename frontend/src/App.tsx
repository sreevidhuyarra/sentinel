import { lazy, Suspense } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { Loading } from "./components/ui";

// One chunk per page: the chart library loads only with the pages that draw charts.
const Overview = lazy(() => import("./pages/Overview"));
const Alerts = lazy(() => import("./pages/Alerts"));
const AlertDetail = lazy(() => import("./pages/AlertDetail"));
const Report = lazy(() => import("./pages/Report"));
const Phishing = lazy(() => import("./pages/Phishing"));
const ModelHealth = lazy(() => import("./pages/ModelHealth"));
const Robustness = lazy(() => import("./pages/Robustness"));

const NAV = [
  ["/", "Live overview"],
  ["/alerts", "Alerts"],
  ["/phishing", "Phishing check"],
  ["/health", "Model health"],
  ["/robustness", "Robustness lab"],
] as const;

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <span className="font-semibold">Sentinel</span>
          <nav className="flex flex-wrap gap-1 text-sm">
            {NAV.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"}
                       className={({ isActive }) => `rounded-md px-2.5 py-1 ${isActive ? "bg-ink text-surface" : "text-ink-2 hover:bg-page"}`}>
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-5">
        <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/alerts/:id" element={<AlertDetail />} />
          <Route path="/reports/:id" element={<Report />} />
          <Route path="/phishing" element={<Phishing />} />
          <Route path="/health" element={<ModelHealth />} />
          <Route path="/robustness" element={<Robustness />} />
          <Route path="*" element={<p className="text-sm text-ink-2">Page not found.</p>} />
        </Routes>
        </Suspense>
      </main>
    </div>
  );
}
