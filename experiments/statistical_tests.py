"""Statistical comparison helpers for completed experiment runs."""

from __future__ import annotations

from scipy.stats import wilcoxon


def paired_wilcoxon(values_a, values_b):
    """Return a paired Wilcoxon signed-rank test when enough pairs exist."""

    pairs = [(a, b) for a, b in zip(values_a, values_b) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    a, b = zip(*pairs)
    statistic, p_value = wilcoxon(a, b)
    return {"statistic": float(statistic), "p_value": float(p_value), "n_pairs": len(pairs)}
