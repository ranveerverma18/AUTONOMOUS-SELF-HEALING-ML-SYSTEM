import { useMemo } from 'react';

export const chartAnimation = {
  isAnimationActive: true,
  animationDuration: 420,
  animationEasing: "ease-out",
};

export function getLastPoint(data) {
  return data?.length ? data[data.length - 1] : null;
}

/**
 * Hook to build fixed XAxis props for a rolling time-series chart.
 * Uses useMemo to prevent generating new array references on every hover/re-render,
 * which previously caused massive CPU spikes in Recharts.
 */
export function useFixedXAxisProps(data, numTicks = 7) {
  return useMemo(() => {
    if (!data?.length) {
      return {
        dataKey: "sampleIndex",
        type: "number",
        domain: [0, 1],
        tick: { fill: "#94a3b8", fontSize: 11 },
      };
    }

    const n = data.length;
    const firstIndex = data[0].sampleIndex;
    const lastIndex = data[n - 1].sampleIndex;

    const timeMap = new Map(data.map((p) => [p.sampleIndex, p.time]));

    const clamp = (v) => Math.max(0, Math.min(n - 1, v));
    const ticks = Array.from({ length: numTicks }, (_, i) => {
      const dataIdx = clamp(Math.round((i / (numTicks - 1)) * (n - 1)));
      return data[dataIdx].sampleIndex;
    });

    const uniqueTicks = [...new Set(ticks)];

    function tickFormatter(value) {
      return timeMap.get(value) ?? "";
    }

    return {
      dataKey: "sampleIndex",
      type: "category",
      ticks: uniqueTicks,
      tickFormatter,
      tick: { fill: "#94a3b8", fontSize: 11 },
      interval: 0,
    };
  }, [data, numTicks]);
}
