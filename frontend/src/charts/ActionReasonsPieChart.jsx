import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

const COLORS = ["#22d3ee", "#f59e0b", "#fb7185", "#a78bfa", "#34d399", "#f97316"];

function ActionReasonsTooltip({ active, data }) {
  if (!active || !data?.length) return null;

  return (
    <div className="chart-tooltip action-reasons-tooltip">
      <p className="chart-tooltip-title">Action events</p>
      {data.map((entry, index) => (
        <div className="chart-tooltip-row" key={entry.reason}>
          <span
            className="chart-tooltip-swatch"
            style={{ backgroundColor: COLORS[index % COLORS.length] }}
          />
          <span>{entry.reason}</span>
          <strong>{entry.value}</strong>
        </div>
      ))}
    </div>
  );
}

export default function ActionReasonsPieChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="reason"
          cx="50%"
          cy="50%"
          innerRadius={55}
          outerRadius={95}
        >
          {data.map((entry, idx) => (
            <Cell key={entry.reason} fill={COLORS[idx % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          content={({ active }) => <ActionReasonsTooltip active={active} data={data} />}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
