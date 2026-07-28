import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { chartAnimation, useFixedXAxisProps } from "./chartUtils";

export default function AccuracyErrorChart({ data }) {
  const xAxisProps = useFixedXAxisProps(data);
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ left: 8, right: 30, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.35)" />
        <XAxis {...xAxisProps} />
        <YAxis domain={[0, 1]} width={45} tick={{ fill: "#94a3b8", fontSize: 11 }} />
        <Tooltip labelFormatter={(value) => `${xAxisProps.tickFormatter?.(value) ?? ""}`} />
        <Legend />
        <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke="#34d399" strokeWidth={2.3} dot={false} connectNulls {...chartAnimation} />
        <Line type="monotone" dataKey="error" name="Error" stroke="#f87171" strokeWidth={2.3} dot={false} connectNulls {...chartAnimation} />
      </LineChart>
    </ResponsiveContainer>
  );
}
