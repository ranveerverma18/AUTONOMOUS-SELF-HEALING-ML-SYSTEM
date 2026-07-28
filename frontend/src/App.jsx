import { useEffect, useMemo, useRef, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AnomalyModal from "./components/AnomalyModal";
import AppLayout from "./layout/AppLayout";
import DriftMonitoringPage from "./pages/DriftMonitoringPage";
import RateControlPage from "./pages/RateControlPage";
import SelfHealingGovernancePage from "./pages/SelfHealingGovernancePage";
import SystemOverviewPage from "./pages/SystemOverviewPage";
import {
  EVENT_TYPES,
  getDashboard,
  injectAnomalies,
  resetRuntime,
  updateControls,
} from "./services/api";

const DASHBOARD_POLL_MS = 500;

export default function App() {
  const [theme, setTheme] = useState("dark");
  const [dashboardData, setDashboardData] = useState(null);
  const [controls, setControls] = useState(null);
  const [connectionState, setConnectionState] = useState("connecting");
  const [isInjecting, setIsInjecting] = useState(false);
  const [scenarioModalOpen, setScenarioModalOpen] = useState(false);
  const mountedRef = useRef(false);
  const dashboardRequestInFlight = useRef(false);
  const dashboardRequestPromise = useRef(null);
  const pendingControlsRef = useRef(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  async function refreshDashboard({ queueAfterCurrent = false } = {}) {
    if (dashboardRequestInFlight.current) {
      if (queueAfterCurrent && dashboardRequestPromise.current) {
        await dashboardRequestPromise.current.catch(() => {});
        return refreshDashboard();
      }
      return null;
    }

    dashboardRequestInFlight.current = true;
    const requestPromise = (async () => {
      try {
        const data = await getDashboard();
        if (mountedRef.current) {
          setDashboardData(data);
          setControls({
            simulatedRate: data.rateControl.simulatedRate,
            rateLimit: data.rateControl.rateLimit,
            rateLimitEnabled: data.rateControl.rateLimitEnabled,
          });
          setConnectionState("live");
        }
        return data;
      } catch (error) {
        if (mountedRef.current) {
          setConnectionState("offline");
        }
        throw error;
      } finally {
        dashboardRequestInFlight.current = false;
        if (dashboardRequestPromise.current === requestPromise) {
          dashboardRequestPromise.current = null;
        }
      }
    })();

    dashboardRequestPromise.current = requestPromise;
    return requestPromise;
  }

  useEffect(() => {
    mountedRef.current = true;

    refreshDashboard().catch(() => {});
    const intervalId = window.setInterval(() => {
      if (document.visibilityState === "hidden") {
        return;
      }
      refreshDashboard().catch(() => {});
    }, DASHBOARD_POLL_MS);

    return () => {
      mountedRef.current = false;
      window.clearInterval(intervalId);
    };
  }, []);

  async function handleInjectAnomalies(payload) {
    setIsInjecting(true);
    try {
      await injectAnomalies(payload);
      await refreshDashboard({ queueAfterCurrent: true });
    } finally {
      setIsInjecting(false);
    }
  }

  function handleControlsChange(nextControls) {
    setControls((prev) => ({ ...prev, ...nextControls }));
    pendingControlsRef.current = { ...(pendingControlsRef.current ?? {}), ...nextControls };
  }

  async function handleControlsCommit() {
    const pending = pendingControlsRef.current;
    if (!pending) return;
    pendingControlsRef.current = null;
    try {
      await updateControls(pending);
      await refreshDashboard({ queueAfterCurrent: true });
    } catch {
      setConnectionState("offline");
    }
  }

  async function handleResetRuntime() {
    await resetRuntime();
    await refreshDashboard({ queueAfterCurrent: true });
  }

  const outletContext = useMemo(
    () => ({
      dashboardData,
      controls,
      eventTypes: EVENT_TYPES,
      connectionState,
      isInjecting,
      onInjectAnomalies: handleInjectAnomalies,
      onControlsChange: handleControlsChange,
      onControlsCommit: handleControlsCommit,
      onResetRuntime: handleResetRuntime,
    }),
    [dashboardData, controls, connectionState, isInjecting]
  );

  return (
    <BrowserRouter>
      <AnomalyModal
        isOpen={scenarioModalOpen}
        onClose={() => setScenarioModalOpen(false)}
        onInject={handleInjectAnomalies}
        isInjecting={isInjecting}
      />
      <Routes>
        <Route
          path="/"
          element={
            <AppLayout
              theme={theme}
              onThemeToggle={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
              modelVersion={dashboardData?.overview?.activeModelVersion}
              connectionState={connectionState}
              anomaly={dashboardData?.overview?.scenario}
              isInjecting={isInjecting}
              onOpenScenarioModal={() => setScenarioModalOpen(true)}
              outletContext={outletContext}
            />
          }
        >
          <Route index element={<SystemOverviewPage />} />
          <Route path="drift-monitoring" element={<DriftMonitoringPage />} />
          <Route path="rate-control" element={<RateControlPage />} />
          <Route path="self-healing" element={<SelfHealingGovernancePage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
