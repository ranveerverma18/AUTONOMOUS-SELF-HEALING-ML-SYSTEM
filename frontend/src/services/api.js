export const EVENT_TYPES = ["DRIFT", "RATE_LIMIT", "MODEL", "OVERLOAD", "CONFIDENCE"];

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }
  return response.json();
}

export function getDashboard() {
  return request("/dashboard");
}

export function fetchScenarios() {
  return request("/scenarios");
}

export function injectAnomalies(payload = {}) {
  return request("/anomalies", {
    method: "POST",
    body: JSON.stringify({ scenario: "sudden_spike", ...payload }),
  });
}

export function updateControls(payload) {
  return request("/controls", { method: "POST", body: JSON.stringify(payload) });
}

export function resetRuntime() {
  return request("/reset", { method: "POST" });
}
