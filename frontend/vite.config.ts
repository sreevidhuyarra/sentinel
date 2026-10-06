import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The API serves the built app under /app (FastAPI), so client routes never collide with API
// routes such as /alerts/{id}. In development, Vite proxies API calls to the running API.
const api = "http://127.0.0.1:8000";
const proxied = ["/alerts", "/stats", "/mlops", "/robustness", "/copilot", "/reports", "/score", "/models", "/health"];

export default defineConfig({
  base: "/app/",
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      ...Object.fromEntries(proxied.map((p) => [p, api])),
      "/ws": { target: api.replace("http", "ws"), ws: true },
    },
  },
});
