import { useMemo } from "react";
import { useOutletContext } from "react-router-dom";
import StreamBacklogChart from "../charts/StreamBacklogChart";
import RateGaugeChart from "../charts/RateGaugeChart";
import RateLimitChart from "../charts/RateLimitChart";
import AuditLogTable from "../components/AuditLogTable";
import MetricCard from "../components/MetricCard";
import SectionCard from "../components/SectionCard";

export default function RateControlPage() {
  const { dashboardData, controls, connectionState, onControlsChange, onControlsCommit } = useOutletContext();
  const controlsDisabled = connectionState !== "live";
  const rateLimitLogs = useMemo(
    () => (dashboardData?.auditLogs ?? []).filter((log) => log.eventType === "RATE_LIMIT"),
    [dashboardData?.auditLogs]
  );

  if (!dashboardData?.rateControl || !controls) {
    return (
      <div className="page-stack">
        <SectionCard
          title={connectionState === "offline" ? "Backend unavailable" : "Waiting for rate-control metrics"}
          subtitle="Rate control appears after the backend publishes processed telemetry."
        >
          <div className="inset-card">
            <p className="inset-text">
              {connectionState === "offline"
                ? "No processed rate-control snapshot is available."
                : "Live rate-control metrics are still loading."}
            </p>
          </div>
        </SectionCard>
      </div>
    );
  }

  const rateControl = dashboardData.rateControl;
  const activeProcessingLimit = rateControl.appliedRateLimit ?? rateControl.workerCapacity ?? controls.rateLimit;
  const gaugeMax = Math.max(activeProcessingLimit, rateControl.currentRate ?? 0, 1);
  const backlogCapacity = rateControl.queueCapacity ?? 500;
  const loadSheddingActive =
    rateControl.loadSheddingActive ||
    rateControl.loadSheddingRecent > 0 ||
    rateControl.streamBacklog >= backlogCapacity;

  return (
    <div className="page-stack">
      <div className="dashboard-grid dashboard-grid-3">
        <SectionCard
          title="Processed Throughput"
          subtitle="Events admitted into the ML inference pipeline"
        >
          <RateGaugeChart
            value={rateControl.currentRate}
            max={gaugeMax}
            capacityLabel="active limit"
          />
        </SectionCard>

        <SectionCard
          title="Rate Controls"
          subtitle="Operator controls for stream pressure and processing limits"
        >
          <div className="control-stack">
            <div>
              <div className="control-row">
                <span>Simulated incoming rate</span>
                <span className="mono-text accent-cyan">{controls.simulatedRate} eps</span>
              </div>
              <input
                type="range"
                min={1}
                max={30}
                step={1}
                value={controls.simulatedRate}
                disabled={controlsDisabled}
                onChange={(event) =>
                  onControlsChange({ simulatedRate: Number(event.target.value) })
                }
                onPointerUp={onControlsCommit}
                onMouseUp={onControlsCommit}
                className="range-input"
              />
            </div>

            <div className="inset-card">
              <label className="toggle-row">
                <span>Enable rate limiting</span>
                <button
                  type="button"
                  disabled={controlsDisabled}
                  onClick={() => {
                    onControlsChange({ rateLimitEnabled: !controls.rateLimitEnabled });
                    onControlsCommit();
                  }}
                  className={`toggle-pill ${controls.rateLimitEnabled ? "is-on" : "is-off"}`}
                  title="Toggle rate limiting"
                >
                  <span
                    className={`toggle-knob ${controls.rateLimitEnabled ? "is-on" : "is-off"}`}
                  />
                </button>
              </label>
            </div>

            <div>
              <div className="control-row">
                <span>Configured processing limit</span>
                <span className="mono-text accent-amber">{controls.rateLimit} eps</span>
              </div>
              <input
                type="range"
                min={1}
                max={40}
                step={1}
                disabled={controlsDisabled}
                value={controls.rateLimit}
                onChange={(event) =>
                  onControlsChange({ rateLimit: Number(event.target.value) })
                }
                onPointerUp={onControlsCommit}
                onMouseUp={onControlsCommit}
                className="range-input"
              />
            </div>
          </div>
        </SectionCard>

        <div className="dashboard-grid dashboard-grid-single-gap">
          <MetricCard
            label="Applied Processing Limit"
            value={`${rateControl.appliedRateLimit} eps`}
            hint={rateControl.controllerState}
          />
          <MetricCard
            label="Queued Backlog"
            value={`${rateControl.streamBacklog}/${backlogCapacity}`}
            hint={
              loadSheddingActive
                ? `Load shedding active: ${rateControl.loadSheddingTotal ?? 0} stale packets discarded`
                : `${rateControl.throttledRate} eps held back`
            }
            status={loadSheddingActive ? "Critical" : undefined}
          />
        </div>
      </div>

      <div className="dashboard-grid dashboard-grid-3">
        <MetricCard
          label="Incoming Rate"
          value={`${rateControl.incomingRate} eps`}
          hint="Events offered to ingestion"
        />
        <MetricCard
          label="Processed Rate"
          value={`${rateControl.processedRate} eps`}
          hint="Events consumed by ML workers"
        />
        <MetricCard
          label="Overload Risk"
          value={`${rateControl.overloadRisk}%`}
          hint={rateControl.controllerReason}
        />
      </div>

      <div className="dashboard-grid dashboard-grid-2">
        <SectionCard
          title="Ingress, Processing, and Limits"
          subtitle="Adaptive controller behavior over time"
        >
          <RateLimitChart
            data={rateControl.actualVsLimitSeries}
            enabled={rateControl.rateLimitEnabled}
          />
        </SectionCard>

        <SectionCard
          title="Stream Backlog"
          subtitle="Queued events waiting for processing capacity"
        >
          <StreamBacklogChart data={rateControl.streamBacklogSeries} />
        </SectionCard>
      </div>

      <div className="dashboard-grid dashboard-grid-2">
        <MetricCard
          label="CPU Usage"
          value={`${rateControl.cpuUsage}%`}
          hint="Streaming workers"
        />
        <MetricCard
          label="Memory Usage"
          value={`${rateControl.memoryUsage}%`}
          hint="Inference and buffering"
        />
      </div>

      <SectionCard
        title="Ingestion Control Audit Logs"
        subtitle="Rate-control policy updates, throttling, and shedding events"
        className="governance-log-card"
      >
        <AuditLogTable logs={rateLimitLogs} />
      </SectionCard>
    </div>
  );
}
