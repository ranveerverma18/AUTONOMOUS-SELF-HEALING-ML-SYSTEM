import {
  PolarAngleAxis,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
} from "recharts";

export default function RateGaugeChart({ value, max = 40, capacityLabel = "active limit" }) {
  const displayValue = Number(value ?? 0);
  const safeMax = Math.max(Number(max) || 1, 1);
  const safeValue = Math.max(0, Math.min(displayValue, safeMax));
  const percent = Number(((displayValue / safeMax) * 100).toFixed(1));

  return (
    <div className="rate-gauge">
      <ResponsiveContainer width="100%" height="100%">
        <RadialBarChart
          data={[{ name: "Processed throughput", value: safeValue, fill: "#22d3ee" }]}
          startAngle={210}
          endAngle={-30}
          innerRadius="68%"
          outerRadius="92%"
          barSize={18}
        >
          <PolarAngleAxis type="number" domain={[0, safeMax]} tick={false} />
          <RadialBar
            dataKey="value"
            cornerRadius={8}
            background
            isAnimationActive
            animationDuration={420}
            animationEasing="ease-out"
          />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="rate-gauge-overlay">
        <p className="rate-gauge-value">{displayValue.toFixed(1)}</p>
        <p className="rate-gauge-label">events/sec</p>
        <p className="rate-gauge-meta">{percent}% of {capacityLabel}</p>
      </div>
    </div>
  );
}
