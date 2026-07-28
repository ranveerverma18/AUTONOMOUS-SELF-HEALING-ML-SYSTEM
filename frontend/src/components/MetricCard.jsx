import { Activity } from "lucide-react";

const statusStyles = {
  Healthy: "status-chip status-healthy",
  Warning: "status-chip status-warning",
  Critical: "status-chip status-critical",
};

const cardStatusStyles = {
  Healthy: "status-card status-card-healthy",
  Warning: "status-card status-card-warning",
  Critical: "status-card status-card-critical",
};

export default function MetricCard({ label, value, hint, status }) {
  const cardStatusClass = status ? cardStatusStyles[status] || "" : "";

  return (
    <article className={`panel-card metric-card ${cardStatusClass}`.trim()}>
      <div className="metric-head">
        <p className="metric-label">{label}</p>
        {status ? (
          <span
            className={`metric-status ${statusStyles[status] || "status-chip status-neutral"}`}
          >
            <Activity size={12} /> {status}
          </span>
        ) : null}
      </div>
      <h3 className="metric-value">{value}</h3>
      <p className="metric-hint">{hint}</p>
    </article>
  );
}
