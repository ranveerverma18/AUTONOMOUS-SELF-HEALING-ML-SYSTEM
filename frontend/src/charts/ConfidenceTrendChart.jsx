import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { chartAnimation, useFixedXAxisProps } from "./chartUtils";

export default function ConfidenceTrendChart({ data }) {
  const xAxisProps = useFixedXAxisProps(data);
  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={data} margin={{ left: 8, right: 30, top: 8, bottom: 8 }}>
        <defs>
          <linearGradient id="confidenceFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.45} />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.08} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.35)" />
        <XAxis {...xAxisProps} />
        <YAxis width={45} tick={{ fill: "#94a3b8", fontSize: 11 }} />
        <Tooltip labelFormatter={(value) => `${xAxisProps.tickFormatter?.(value) ?? ""}`} />
        <Area type="monotone" dataKey="confidence" name="Prediction confidence" stroke="#38bdf8" fill="url(#confidenceFill)" strokeWidth={2.3} connectNulls {...chartAnimation} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
