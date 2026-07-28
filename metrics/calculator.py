"""
metrics/calculator.py - Dashboard metric helpers.

Pure-function utilities that transform raw runtime state (prediction
errors, drift scores, confidence values, feature windows) into the
structured payloads consumed by the frontend dashboard panels.

All functions are stateless and side-effect-free – they receive data
as arguments and return plain Python scalars, lists, or dicts.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def confidence_value(pred_std: float, error: float) -> float:
    """
    Map prediction standard deviation and absolute error to a [0, 1]
    confidence score for the dashboard.

    Args:
        pred_std : Standard deviation across RF tree predictions.
        error    : Absolute prediction error (|actual - pred|).

    Returns:
        Clipped confidence in [0, 1].
    """
    return float(np.clip(1.0 - (pred_std / 95.0) - (error / 320.0), 0.0, 1.0))


def status_from_metrics(drift_score: float, rolling_avg: float | None, action: str) -> str:
    """
    Derive a traffic-light status string from the current drift score,
    rolling MAE, and decision-engine action.

    Returns:
        'Critical' | 'Warning' | 'Healthy'
    """
    if action in {"RETRAIN_URGENT", "ALERT"} or drift_score >= 0.82:
        return "Critical"
    if action in {"RETRAIN", "MONITOR", "WATCH"} or drift_score >= 0.48:
        return "Warning"
    if rolling_avg is not None and rolling_avg > 70:
        return "Warning"
    return "Healthy"


# ---------------------------------------------------------------------------
# Feature-level helpers
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)] + [
    f"sensor_{i}" for i in range(1, 22)
]

_DEFAULT_DISPLAY_FEATURES = [
    "sensor_2", "sensor_4", "sensor_7", "sensor_11", "sensor_15", "sensor_20"
]


def feature_scores(
    current_feature_window: deque,
    baseline_means: dict[str, float],
    baseline_stds: dict[str, float],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    top_n: int = 8,
) -> list[dict]:
    """
    Compute per-feature drift scores based on mean deviation from the
    training baseline, normalised to [0, 1].

    Args:
        current_feature_window : Ring-buffer of recent feature dicts.
        baseline_means         : Training-set per-feature means.
        baseline_stds          : Training-set per-feature std-devs (min 0.5).
        feature_columns        : Feature column names to evaluate.
        top_n                  : Return only the *top_n* highest-scoring features.

    Returns:
        Sorted list of ``{"feature": str, "score": float}`` dicts.
    """
    if not current_feature_window:
        return [{"feature": f, "score": 0.0} for f in _DEFAULT_DISPLAY_FEATURES]

    scores = []
    for feature in feature_columns:
        values = [row[feature] for row in current_feature_window]
        bm = baseline_means.get(feature, 0.0)
        bs = max(baseline_stds.get(feature, 1.0), 0.5)
        score = min(abs(float(np.mean(values)) - bm) / (3.0 * bs), 1.0)
        scores.append({"feature": feature, "score": round(float(score), 3)})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_n]


def distribution_shift(
    feature_scores_list: list[dict],
    current_feature_window: deque,
    baseline_means: dict[str, float],
    baseline_stds: dict[str, float],
    top_n: int = 6,
) -> list[dict]:
    """
    Build the distribution-shift payload for the dashboard, showing
    where each of the top-drifting features sits relative to its
    training baseline (normalised to a [0, 1] range around 0.5).

    Args:
        feature_scores_list    : Output of :func:`feature_scores`.
        current_feature_window : Ring-buffer of recent feature dicts.
        baseline_means         : Training-set per-feature means.
        baseline_stds          : Training-set per-feature std-devs.
        top_n                  : Number of features to include.

    Returns:
        List of ``{"feature": str, "baseline": 0.5, "current": float}`` dicts.
    """
    output = []
    for item in feature_scores_list[:top_n]:
        feature = item["feature"]
        values = [row[feature] for row in current_feature_window]
        bm = baseline_means.get(feature, 0.0)
        bs = max(baseline_stds.get(feature, 1.0), 0.5)
        current_mean = float(np.mean(values)) if values else bm
        output.append(
            {
                "feature": feature,
                "baseline": 0.5,
                "current": round(
                    float(np.clip(0.5 + (current_mean - bm) / (6 * bs), 0, 1)), 3
                ),
            }
        )
    return output


# ---------------------------------------------------------------------------
# Confidence histogram
# ---------------------------------------------------------------------------

def confidence_histogram(confidences: deque) -> list[dict]:
    """
    Build a 10-bucket histogram of recent confidence values for the
    dashboard Confidence Distribution panel.

    Args:
        confidences : Ring-buffer of recent [0, 1] confidence floats.

    Returns:
        List of ``{"bucket": str, "count": int}`` dicts for buckets
        [0.00-0.10), [0.10-0.20), ..., [0.90-1.00].
    """
    values = list(confidences)
    buckets = [
        (index / 10, (index + 1) / 10, f"{index / 10:.2f}-{(index + 1) / 10:.2f}")
        for index in range(10)
    ]
    return [
        {
            "bucket": bucket,
            "count": sum(
                1
                for v in values
                if (low <= v <= high if high == 1.0 else low <= v < high)
            ),
        }
        for low, high, bucket in buckets
    ]
