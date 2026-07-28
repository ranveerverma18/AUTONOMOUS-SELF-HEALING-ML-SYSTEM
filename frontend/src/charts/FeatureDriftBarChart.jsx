import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function FeatureDriftBarChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.35)" />
        <XAxis dataKey="feature" tick={{ fill: "#94a3b8", fontSize: 11 }} />
        <YAxis domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
        <Tooltip />
        <Bar dataKey="score" name="Feature drift score" fill="#a855f7" radius={[6, 6, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
