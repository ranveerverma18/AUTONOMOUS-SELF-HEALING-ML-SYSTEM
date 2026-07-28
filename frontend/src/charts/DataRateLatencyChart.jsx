import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import LiveChartFrame from "./LiveChartFrame";
import { chartAnimation, getLastPoint, useFixedXAxisProps } from "./chartUtils";

export default function DataRateLatencyChart({ data }) {
  const last = getLastPoint(data);
  const xAxisProps = useFixedXAxisProps(data);

  return (
    <LiveChartFrame points={data?.length ?? 0}>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.35)" />
          <XAxis {...xAxisProps} />
          <YAxis yAxisId="left" domain={[0, 60]} width={45} tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <YAxis yAxisId="right" orientation="right" domain={[0, 200]} width={45} tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <Tooltip labelFormatter={(value) => `${xAxisProps.tickFormatter?.(value) ?? ""}`} />
          <Legend />
          <Line yAxisId="left" type="monotone" dataKey="dataRate" name="Ingestion rate (eps)" stroke="#22d3ee" strokeWidth={2.6} dot={false} connectNulls {...chartAnimation} />
          <Line yAxisId="right" type="monotone" dataKey="latency" name="Latency (ms)" stroke="#f59e0b" strokeWidth={2.4} dot={false} connectNulls {...chartAnimation} />
          {last ? <ReferenceDot yAxisId="left" x={last.sampleIndex} y={last.dataRate} r={4} fill="#22d3ee" stroke="#0f172a" /> : null}
        </LineChart>
      </ResponsiveContainer>
    </LiveChartFrame>
  );
}
