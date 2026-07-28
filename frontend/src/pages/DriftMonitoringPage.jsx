import { useOutletContext } from "react-router-dom";
import AccuracyErrorChart from "../charts/AccuracyErrorChart";
import ConfidenceTrendChart from "../charts/ConfidenceTrendChart";
import DistributionShiftChart from "../charts/DistributionShiftChart";
import FeatureDriftBarChart from "../charts/FeatureDriftBarChart";
import AlertsTimeline from "../components/AlertsTimeline";
import SectionCard from "../components/SectionCard";

export default function DriftMonitoringPage() {
  const { dashboardData, connectionState } = useOutletContext();

  if (!dashboardData?.drift) {
    return (
      <div className="page-stack">
        <SectionCard
          title={connectionState === "offline" ? "Backend unavailable" : "Waiting for drift metrics"}
          subtitle="Drift monitoring appears after the backend publishes processed telemetry."
        >
          <div className="inset-card">
            <p className="inset-text">
              {connectionState === "offline"
                ? "No processed drift snapshot is available."
                : "Live drift metrics are still loading."}
            </p>
          </div>
        </SectionCard>
      </div>
    );
  }

  return (
    <div className="page-stack">
      <div className="dashboard-grid dashboard-grid-2">
        <SectionCard
          title="Feature Distribution Shift"
          subtitle="Baseline distribution compared with the current feature window"
        >
          <DistributionShiftChart data={dashboardData.drift.featureDistributionShift} />
        </SectionCard>

        <SectionCard
          title="Drift Score Per Feature"
          subtitle="Highest contributing feature-level drift signals"
        >
          <FeatureDriftBarChart data={dashboardData.drift.featureScores} />
        </SectionCard>
      </div>

      <div className="dashboard-grid dashboard-grid-2">
        <SectionCard
          title="Model Error Drift"
          subtitle="Accuracy and prediction error over time"
        >
          <AccuracyErrorChart data={dashboardData.drift.conceptSeries} />
        </SectionCard>

        <SectionCard
          title="Prediction Confidence"
          subtitle="Confidence stability across recent predictions"
        >
          <ConfidenceTrendChart data={dashboardData.drift.conceptSeries} />
        </SectionCard>
      </div>

      <SectionCard
        title="Operational Alerts"
        subtitle="Latest drift and confidence-related events"
      >
        <AlertsTimeline alerts={dashboardData.drift.alerts} />
      </SectionCard>
    </div>
  );
}
