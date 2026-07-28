import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function ConfidenceHistogramChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.35)" />
        <XAxis dataKey="bucket" tick={{ fill: "#94a3b8", fontSize: 11 }} />
        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
        <Tooltip />
        <Bar
          dataKey="count"
          name="Prediction count"
          fill="#22c55e"
          radius={[6, 6, 0, 0]}
          isAnimationActive
          animationDuration={420}
          animationEasing="ease-out"
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
