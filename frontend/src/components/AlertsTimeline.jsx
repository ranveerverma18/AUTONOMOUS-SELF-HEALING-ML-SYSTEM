import StatusPill from "./StatusPill";

export default function AlertsTimeline({ alerts }) {
  if (!alerts?.length) {
    return (
      <div className="inset-card">
        <p className="inset-text">No drift or confidence alerts have been raised.</p>
      </div>
    );
  }

  return (
    <ul className="timeline-list">
      {alerts.map((alert) => (
        <li
          key={`${alert.timestamp}-${alert.message}`}
          className="timeline-item"
        >
          <div className="timeline-head">
            <span className="mono-text timeline-time">{alert.timestamp}</span>
            <StatusPill value={alert.severity} />
          </div>
          <p className="timeline-message">{alert.message}</p>
        </li>
      ))}
    </ul>
  );
}
