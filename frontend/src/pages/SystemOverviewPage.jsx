import { useOutletContext } from "react-router-dom";
import ConfidenceHistogramChart from "../charts/ConfidenceHistogramChart";
import DataRateLatencyChart from "../charts/DataRateLatencyChart";
import DriftScoreLineChart from "../charts/DriftScoreLineChart";
import MetricCard from "../components/MetricCard";
import SectionCard from "../components/SectionCard";
import StatusPill from "../components/StatusPill";

export default function SystemOverviewPage() {
  const { dashboardData, connectionState } = useOutletContext();

  if (!dashboardData?.overview) {
    return (
      <div className="page-stack">
        <SectionCard
          title={connectionState === "offline" ? "Backend unavailable" : "Waiting for processed telemetry"}
          subtitle="Operational metrics appear after the backend publishes a dashboard snapshot."
        >
          <div className="inset-card">
            <p className="inset-text">
              {connectionState === "offline"
                ? "The backend did not return a dashboard snapshot."
                : "Live operational data is still loading."}
            </p>
          </div>
        </SectionCard>
      </div>
    );
  }

  const latest = dashboardData.overview.series.at(-1);
  const latestSample = dashboardData.overview.latestSample;
  const scenario = dashboardData.overview.scenario;
  const meta = dashboardData.overview.meta;

  const adwinLabel = latestSample.adwinDrift ? "Concept drift" : "ADWIN stable";
  const ksLabel = latestSample.dataDrift ? "Feature drift" : "KS stable";
  const anomalyLabel = latestSample.anomalyDetected ? "Anomaly detected" : "Anomaly clear";
  const lastUpdated = meta?.lastUpdatedAt?.split(" ").at(-1) ?? latest.time;

  return (
    <div className="page-stack">
      {scenario?.active && (
        <div className={`scenario-banner scenario-banner--${(scenario.severity ?? "medium").toLowerCase()}`}>
          <span className="scenario-banner-icon">!</span>
          <span>
            <strong>{scenario.name}</strong> scenario active -{" "}
            <span className="mono-text">{scenario.remaining} cycles remaining</span>
          </span>
        </div>
      )}

      <div className="live-ops-strip panel-card">
        <div>
          <span className={`live-chart-badge ${connectionState === "live" ? "live-chart-badge-cyan" : "live-chart-badge-amber"}`}>
            <span className="live-chart-dot" />
            {connectionState === "live" ? "Streaming" : "Reconnecting"}
          </span>
        </div>
        <div className="live-ops-item">
          <span>Processing tick</span>
          <strong>{meta?.workerTickMs ?? 50} ms</strong>
        </div>
        <div className="live-ops-item">
          <span>Snapshot</span>
          <strong>{meta?.snapshotIntervalMs ?? 500} ms</strong>
        </div>
        <div className="live-ops-item">
          <span>Last update</span>
          <strong>{lastUpdated}</strong>
        </div>
      </div>

      <div className="dashboard-grid dashboard-grid-3">
        <MetricCard label="System Status" value={dashboardData.overview.systemStatus} hint="Composite health signal" status={dashboardData.overview.systemStatus} />
        <MetricCard label="Ingestion Rate" value={`${latest.dataRate} eps`} hint="Current event ingress" />
        <MetricCard label="Drift Score" value={latest.driftScore} hint="Combined KS and ADWIN signal" />
        <MetricCard label="Active Model Version" value={dashboardData.overview.activeModelVersion} hint="Serving model in production" />
        <MetricCard label="Latency" value={`${latest.latency} ms`} hint="End-to-end processing latency" />
        <MetricCard label="Stream Backlog" value={`${latest.streamBacklog}`} hint="Unprocessed backlog count" />
      </div>

      <div className="dashboard-grid dashboard-grid-3">
        <SectionCard title="Latest Prediction" subtitle="Most recent turbine RUL inference">
          <div className="live-sample-grid">
            <div>
              <p className="mini-label">Predicted RUL</p>
              <p className="sample-value">{latestSample.prediction}</p>
            </div>
            <div>
              <p className="mini-label">Actual RUL</p>
              <p className="sample-value">{latestSample.actual}</p>
            </div>
            <div>
              <p className="mini-label">Decision</p>
              <StatusPill value={latestSample.action} />
            </div>
          </div>
          <p className="sample-meta">
            Sample #{latestSample.sampleIndex} | interval [{latestSample.interval[0]} - {latestSample.interval[1]}] | MAE: {latest.rollingMae !== null ? latest.rollingMae : "N/A"}
          </p>
          <div className="drift-indicators">
            <span className="drift-indicator-pill">{adwinLabel}</span>
            <span className="drift-indicator-pill">{ksLabel}</span>
            <span className="drift-indicator-pill">{anomalyLabel}</span>
          </div>
        </SectionCard>

        <SectionCard title="Runtime State" subtitle="API and stream execution status">
          <div className="runtime-list">
            <div className="runtime-row">
              <span>Backend</span>
              <StatusPill value={connectionState === "live" ? "Healthy" : "Warning"} />
            </div>
            <div className="runtime-row">
              <span>Model action</span>
              <StatusPill value={latest.action} />
            </div>
            <div className="runtime-row">
              <span>ADWIN drift</span>
              <span className="mono-text">{latest.adwinDrift ? "Detected" : "Clear"}</span>
            </div>
            <div className="runtime-row">
              <span>KS drift</span>
              <span className="mono-text">{latest.dataDrift ? "Detected" : "Clear"}</span>
            </div>
            <div className="runtime-row">
              <span>Cooldown</span>
              <span className="mono-text">
                {latest.cooldownElapsed}/{latest.cooldownRequired}
              </span>
            </div>
          </div>
        </SectionCard>
      </div>

      <div className="dashboard-grid dashboard-grid-2">
        <SectionCard title="Ingestion Rate vs Latency" subtitle="Throughput and processing responsiveness">
          <DataRateLatencyChart data={dashboardData.overview.series} />
        </SectionCard>
        <SectionCard title="Drift Score Over Time" subtitle="Early warning signal for data and concept drift">
          <DriftScoreLineChart data={dashboardData.overview.series} />
        </SectionCard>
      </div>

      <SectionCard title="Prediction Confidence Distribution" subtitle="Histogram of recent confidence scores">
        <ConfidenceHistogramChart data={dashboardData.overview.confidenceHistogram} />
      </SectionCard>
    </div>
  );
}
