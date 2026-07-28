import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { fetchScenarios } from "../services/api";

const SEVERITY_ORDER = ["Critical", "High", "Medium", "Low"];

function SeverityBadge({ severity }) {
  const cls = {
    Critical: "severity-badge--critical",
    High: "severity-badge--high",
    Medium: "severity-badge--medium",
    Low: "severity-badge--low",
  }[severity] ?? "severity-badge--low";
  return <span className={`severity-badge ${cls}`}>{severity}</span>;
}

export default function AnomalyModal({ isOpen, onClose, onInject, isInjecting }) {
  const [scenarios, setScenarios] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [selected, setSelected] = useState(null);
  const overlayRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;

    let cancelled = false;
    setSelected(null);
    setIsLoading(true);
    setLoadError("");

    fetchScenarios()
      .then((data) => {
        if (cancelled) return;

        if (data?.scenarios?.length) {
          const sorted = [...data.scenarios].sort(
            (a, b) => SEVERITY_ORDER.indexOf(a.severity) - SEVERITY_ORDER.indexOf(b.severity)
          );
          setScenarios(sorted);
        } else {
          setScenarios([]);
        }
      })
      .catch(() => {
        if (cancelled) return;
        setScenarios([]);
        setLoadError("Unable to load live scenarios from the backend.");
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;

    const handler = (event) => {
      if (event.key === "Escape") onClose();
    };

    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [isOpen, onClose]);

  function handleOverlayClick(event) {
    if (event.target === overlayRef.current) onClose();
  }

  function handleInject() {
    if (!selected) return;
    onInject({ scenario: selected });
    onClose();
  }

  if (!isOpen) return null;

  const selectedMeta = scenarios.find((scenario) => scenario.id === selected);

  return (
    <div className="modal-overlay" ref={overlayRef} onClick={handleOverlayClick}>
      <div className="modal-panel" role="dialog" aria-modal="true" aria-labelledby="modal-title">
        <div className="modal-header">
          <div className="modal-header-row">
            <div>
              <h2 id="modal-title" className="modal-title">
                Run Fault Scenario
              </h2>
              <p className="modal-subtitle">
                Select a controlled operating condition to validate drift detection,
                rate control, and autonomous model governance.
              </p>
            </div>
            <button className="modal-close-btn" onClick={onClose} aria-label="Close">
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="modal-body">
          <div className="scenario-grid">
            {isLoading ? (
              <div className="scenario-empty-state">Loading operating scenarios...</div>
            ) : scenarios.length ? (
              scenarios.map((scenario) => (
                <button
                  key={scenario.id}
                  type="button"
                  className={`scenario-card${selected === scenario.id ? " selected" : ""}`}
                  onClick={() => setSelected(scenario.id === selected ? null : scenario.id)}
                >
                  <SeverityBadge severity={scenario.severity} />
                  <p className="scenario-name">{scenario.name}</p>
                  <p className="scenario-description">{scenario.description}</p>
                  <div className="scenario-footer">
                    <span className="scenario-tag">{scenario.duration} cycles</span>
                  </div>
                  <div className="scenario-expected">
                    <p className="scenario-expected-label">Expected system response</p>
                    <p className="scenario-expected-text">{scenario.expectedBehavior}</p>
                  </div>
                </button>
              ))
            ) : (
              <div className="scenario-empty-state">
                {loadError || "No operating scenarios are available."}
              </div>
            )}
          </div>
        </div>

        <div className="modal-footer">
          {selectedMeta && (
            <span className="modal-selection-hint">
              Selected: <strong>{selectedMeta.name}</strong> ({selectedMeta.duration} cycles)
            </span>
          )}
          <button type="button" className="secondary-action-btn" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="primary-action-btn"
            disabled={!selected || isInjecting || isLoading}
            onClick={handleInject}
          >
            {isInjecting ? "Running..." : "Run Selected Scenario"}
          </button>
        </div>
      </div>
    </div>
  );
}
