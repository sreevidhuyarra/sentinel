import { useSyncExternalStore } from "react";

// One fixed colour per family (light / dark validated steps). Families that occur together
// in the data were checked all-pairs (Fri: DDoS, PortScan, Bot; Thu: PortScan, WebAttack,
// Infiltration); "Unknown anomaly" is unclassified, so it is neutral grey, not a hue.
const FAMILY_ORDER = ["DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration", "Unknown anomaly"];
const LIGHT = ["#2a78d6", "#eb6834", "#4a3aa7", "#eda100", "#e87ba4", "#1baf7a", "#008300", "#898781"];
const DARK = ["#3987e5", "#d95926", "#9085e9", "#c98500", "#d55181", "#199e70", "#008300", "#898781"];

// Status palette: reserved for severity, never used as a series colour; always shown with a label.
export const SEVERITY_COLOR: Record<string, string> = {
  Critical: "#d03b3b",
  High: "#ec835a",
  Medium: "#fab219",
  Low: "#898781",
};
export const SEVERITIES = ["Critical", "High", "Medium", "Low"];

const query = "(prefers-color-scheme: dark)";
function subscribe(cb: () => void) {
  const m = window.matchMedia(query);
  m.addEventListener("change", cb);
  return () => m.removeEventListener("change", cb);
}

export function useDark(): boolean {
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches, () => false);
}

export function familyColor(family: string, dark: boolean): string {
  const i = FAMILY_ORDER.indexOf(family);
  return (dark ? DARK : LIGHT)[i >= 0 ? i : FAMILY_ORDER.length - 1];
}

export function chartInk(dark: boolean) {
  return dark
    ? { ink: "#ffffff", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", surface: "#1a1a19" }
    : { ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", surface: "#fcfcfb" };
}

export function sortFamilies(names: string[]): string[] {
  return [...names].sort((a, b) => {
    const ia = FAMILY_ORDER.indexOf(a), ib = FAMILY_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
}
