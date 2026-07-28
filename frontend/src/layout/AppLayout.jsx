import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import TopNavbar from "./TopNavbar";

export default function AppLayout({
  theme,
  onThemeToggle,
  modelVersion,
  connectionState,
  anomaly,
  isInjecting,
  onOpenScenarioModal,
  outletContext,
}) {
  return (
    <div className="app-shell">
      <div className="app-frame">
        <Sidebar />
        <main className="app-main">
          <TopNavbar
            theme={theme}
            onThemeToggle={onThemeToggle}
            modelVersion={modelVersion}
            connectionState={connectionState}
            anomaly={anomaly}
            isInjecting={isInjecting}
            onOpenScenarioModal={onOpenScenarioModal}
          />
          <Outlet context={outletContext} />
        </main>
      </div>
    </div>
  );
}
