import { RotateCcw } from "lucide-react";
import { useMemo, useState } from "react";
import { useOutletContext } from "react-router-dom";
import ActionReasonsPieChart from "../charts/ActionReasonsPieChart";
import ModelVersionChart from "../charts/ModelVersionChart";
import AuditLogTable from "../components/AuditLogTable";
import SectionCard from "../components/SectionCard";
import StatusPill from "../components/StatusPill";

const EVENT_TYPE_LABELS = {
  ALL: "All events",
  DRIFT: "Drift",
  RATE_LIMIT: "Rate control",
  MODEL: "Model",
  OVERLOAD: "Overload",
  CONFIDENCE: "Confidence",
};

function parseTimestamp(logTimestamp) {
  return new Date(logTimestamp.replace(" ", "T"));
}

function ShadowEvaluationCard({ shadow }) {
  if (!shadow?.is_evaluating) return null;

  const progress = shadow.samples_needed > 0
    ? Math.min(100, Math.round((shadow.samples_collected / shadow.samples_needed) * 100))
    : 0;

  const prodMae = shadow.production_mae != null ? shadow.production_mae.toFixed(2) : "N/A";
  const shadowMae = shadow.shadow_mae != null ? shadow.shadow_mae.toFixed(2) : "N/A";

  const isBetter = shadow.shadow_mae != null && shadow.production_mae != null
    && shadow.shadow_mae < shadow.production_mae;

  return (
    <div className="shadow-eval-card">
      <div className="shadow-eval-header">
        <span className="shadow-eval-title">Shadow Model Evaluation in Progress</span>
        <span className="shadow-eval-cycles">
          {shadow.samples_collected} / {shadow.samples_needed} cycles
        </span>
      </div>
      <div className="shadow-progress-bar">
        <div className="shadow-progress-fill" style={{ width: `${progress}%` }} />
      </div>
      {(shadow.production_mae != null || shadow.shadow_mae != null) && (
        <div className="shadow-mae-compare">
          <div className="shadow-mae-item">
            <span className="mini-label">Production MAE</span>
            <span className="shadow-mae-value">{prodMae}</span>
          </div>
          <div className="shadow-mae-item">
            <span className="mini-label">Shadow MAE</span>
            <span className={`shadow-mae-value ${isBetter ? "shadow-mae-better" : "shadow-mae-worse"}`}>
              {shadowMae}
              {shadow.shadow_mae != null && shadow.production_mae != null
                ? (isBetter ? " better" : " worse")
                : ""}
            </span>
          </div>
          <div className="shadow-mae-item">
            <span className="mini-label">Outcome</span>
            <span className="mono-text" style={{ fontSize: "0.78rem" }}>
              {shadow.ready_to_decide
                ? (isBetter ? "Ready to promote" : "Will be rejected")
                : "Evaluating..."}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default function SelfHealingGovernancePage() {
  const { dashboardData, eventTypes, onResetRuntime, connectionState } = useOutletContext();
  const [eventFilter, setEventFilter] = useState("ALL");
  const [timeFilter, setTimeFilter] = useState("ALL");

  const shadow = dashboardData?.selfHealing?.shadowEvaluation;
  const auditLogs = dashboardData?.auditLogs ?? [];

  const filteredLogs = useMemo(() => {
    const now = Date.now();
    return auditLogs.filter((log) => {
      const matchesEvent = eventFilter === "ALL" || log.eventType === eventFilter;
      if (!matchesEvent) return false;
      if (timeFilter === "ALL") return true;
      const ts = parseTimestamp(log.timestamp).getTime();
      const diffMinutes = (now - ts) / (1000 * 60);
      if (timeFilter === "1H") return diffMinutes <= 60;
      if (timeFilter === "6H") return diffMinutes <= 360;
      if (timeFilter === "24H") return diffMinutes <= 1440;
      return true;
    });
  }, [auditLogs, eventFilter, timeFilter]);

  if (!dashboardData?.selfHealing) {
    return (
      <div className="page-stack">
        <SectionCard
          title={connectionState === "offline" ? "Backend unavailable" : "Waiting for governance data"}
          subtitle="Model governance appears after the backend publishes processed snapshots."
        >
          <div className="inset-card">
            <p className="inset-text">
              {connectionState === "offline"
                ? "No processed governance snapshot is available."
                : "Live governance data is still loading."}
            </p>
          </div>
        </SectionCard>
      </div>
    );
  }

  return (
    <div className="page-stack">
      {shadow?.is_evaluating && (
        <ShadowEvaluationCard shadow={shadow} />
      )}

      <SectionCard title="Model Deployment History" subtitle="Production model versions over time">
        <ModelVersionChart data={dashboardData.selfHealing.modelHistory} />
      </SectionCard>

      <div className="dashboard-grid dashboard-grid-3">
        <SectionCard title="Autonomous Action Reasons" subtitle="Operational conditions that triggered system actions">
          <ActionReasonsPieChart data={dashboardData.selfHealing.actionReasons} />
        </SectionCard>

        <SectionCard title="Most Recent Action" subtitle="Latest model governance outcome">
          <div className="inset-card">
            <p className="inset-text">{dashboardData.selfHealing.lastActionTaken}</p>
          </div>
        </SectionCard>

        <SectionCard title="Governance State" subtitle="Current model management posture">
          <div className="inset-card">
            <StatusPill value={dashboardData.selfHealing.currentSystemState} />
            {shadow?.is_evaluating && (
              <p className="mini-label" style={{ marginTop: "8px", color: "var(--accent-cyan)" }}>
                Shadow evaluation: {shadow.samples_collected}/{shadow.samples_needed} cycles
              </p>
            )}
          </div>
        </SectionCard>
      </div>

      <SectionCard
        title="Model Governance Audit Logs"
        subtitle="Unified audit trail for model decisions and operational controls"
        className="governance-log-card"
        action={
          <div className="filter-group">
            <button
              type="button"
              className="secondary-action-btn"
              onClick={onResetRuntime}
              title="Restart the streaming runtime"
            >
              <RotateCcw size={14} />
              Reset Runtime
            </button>
            <select
              value={eventFilter}
              onChange={(e) => setEventFilter(e.target.value)}
              className="filter-select"
            >
              <option value="ALL">{EVENT_TYPE_LABELS.ALL}</option>
              {eventTypes.map((type) => (
                <option key={type} value={type}>{EVENT_TYPE_LABELS[type] || type}</option>
              ))}
            </select>
            <select
              value={timeFilter}
              onChange={(e) => setTimeFilter(e.target.value)}
              className="filter-select"
            >
              <option value="ALL">All time</option>
              <option value="1H">Last 1 hour</option>
              <option value="6H">Last 6 hours</option>
              <option value="24H">Last 24 hours</option>
            </select>
          </div>
        }
      >
        <AuditLogTable logs={filteredLogs} />
      </SectionCard>
    </div>
  );
}
