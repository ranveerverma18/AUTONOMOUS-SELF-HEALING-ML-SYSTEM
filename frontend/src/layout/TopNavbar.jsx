import { Activity, Moon, RotateCw, Sun, Zap } from "lucide-react";

export default function TopNavbar({
  theme,
  onThemeToggle,
  modelVersion,
  connectionState,
  anomaly,
  isInjecting,
  onOpenScenarioModal,
}) {
  const isLive = connectionState === "live";
  const connectionLabel =
    connectionState === "live"
      ? "Live"
      : connectionState === "connecting"
        ? "Connecting"
        : "Offline";
  const modelLabel = modelVersion ?? "Waiting for model";

  return (
    <header className="topbar panel-card">
      <div className="topbar-row">
        <div>
          <p className="topbar-kicker">Autonomous Streaming ML Operations</p>
          <h2 className="topbar-title">Industrial IoT Predictive Maintenance</h2>
        </div>
        <div className="topbar-actions">
          <span className={`topbar-badge ${isLive ? "badge-live" : "badge-muted"}`}>
            <Activity size={14} />
            {connectionLabel}
          </span>
          <span className="topbar-badge topbar-hide-mobile">Model {modelLabel}</span>
          {anomaly?.active && (
            <span className="topbar-badge badge-critical topbar-hide-mobile">
              Drift active: {anomaly.remaining} samples
            </span>
          )}
          <button
            type="button"
            onClick={onOpenScenarioModal}
            className="primary-action-btn"
            disabled={isInjecting || !isLive}
            title="Run a controlled fault scenario"
          >
            {isInjecting ? <RotateCw size={15} /> : <Zap size={15} />}
            {isInjecting ? "Running" : "Run Scenario"}
          </button>
          <button
            type="button"
            onClick={onThemeToggle}
            className="topbar-theme-btn"
            title="Toggle color theme"
          >
            {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
            {theme === "dark" ? "Light" : "Dark"}
          </button>
        </div>
      </div>
    </header>
  );
}
