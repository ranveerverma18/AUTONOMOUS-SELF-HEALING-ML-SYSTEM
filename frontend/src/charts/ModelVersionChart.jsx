import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function ModelVersionChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.35)" />
        <XAxis dataKey="time" tick={{ fill: "#94a3b8", fontSize: 11 }} />
        <YAxis allowDecimals={false} tick={{ fill: "#94a3b8", fontSize: 11 }} />
        <Tooltip formatter={(value, name, item) => [item.payload.modelVersion, "Model version"]} />
        <Line type="stepAfter" dataKey="versionIndex" name="Model version" stroke="#06b6d4" strokeWidth={2.8} dot={{ r: 4 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}
