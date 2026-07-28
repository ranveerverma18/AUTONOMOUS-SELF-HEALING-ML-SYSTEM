export default function LiveChartFrame({ children, points = 0, tone = "cyan" }) {
  return (
    <div className="live-chart-frame">
      <div className="live-chart-toolbar">
        <span className={`live-chart-badge live-chart-badge-${tone}`}>
          <span className="live-chart-dot" />
          Live
        </span>
        <span className="live-chart-points">{points} points</span>
      </div>
      {children}
    </div>
  );
}
