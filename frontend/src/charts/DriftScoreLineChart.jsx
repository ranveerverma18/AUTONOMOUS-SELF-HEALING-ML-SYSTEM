import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import LiveChartFrame from "./LiveChartFrame";
import { chartAnimation, getLastPoint, useFixedXAxisProps } from "./chartUtils";

export default function DriftScoreLineChart({ data }) {
  const last = getLastPoint(data);
  const xAxisProps = useFixedXAxisProps(data);

  return (
    <LiveChartFrame points={data?.length ?? 0} tone="rose">
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data} margin={{ left: 8, right: 30, top: 8, bottom: 8 }}>
          <defs>
            <linearGradient id="driftFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#fb7185" stopOpacity={0.46} />
              <stop offset="100%" stopColor="#fb7185" stopOpacity={0.04} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.35)" />
          <XAxis {...xAxisProps} />
          <YAxis domain={[0, 1]} width={45} tick={{ fill: "#94a3b8", fontSize: 11 }} />
          <Tooltip labelFormatter={(value) => `${xAxisProps.tickFormatter?.(value) ?? ""}`} />
          <ReferenceLine y={0.6} stroke="#f43f5e" strokeDasharray="5 5" />
          <Area type="monotone" dataKey="driftScore" name="Drift score" stroke="#fb7185" fill="url(#driftFill)" strokeWidth={2.4} dot={false} connectNulls {...chartAnimation} />
          {last ? <ReferenceDot x={last.sampleIndex} y={last.driftScore} r={4} fill="#fb7185" stroke="#0f172a" /> : null}
        </AreaChart>
      </ResponsiveContainer>
    </LiveChartFrame>
  );
}
