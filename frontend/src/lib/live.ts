import { useEffect, useState } from "react";
import type { Alert } from "./api";

// Live alert feed over WS /ws/alerts; keeps the newest `keep` alerts and reconnects.
export function useLiveAlerts(keep = 50) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: number | undefined;
    let closed = false;
    const open = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws/alerts`);
      ws.onopen = () => setConnected(true);
      ws.onmessage = (e) => {
        const batch = (JSON.parse(e.data).alerts ?? []) as Alert[];
        setAlerts((prev) => [...batch.reverse(), ...prev].slice(0, keep));
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = window.setTimeout(open, 3000);
      };
    };
    open();
    return () => {
      closed = true;
      window.clearTimeout(retry);
      ws?.close();
    };
  }, [keep]);
  return { alerts, connected };
}
