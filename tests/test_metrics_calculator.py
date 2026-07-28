from collections import deque
import pytest
from metrics.calculator import (
    confidence_value,
    status_from_metrics,
    feature_scores,
    distribution_shift,
    confidence_histogram,
)

FEATURE_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)] + [
    f"sensor_{i}" for i in range(1, 22)
]


def test_confidence_value_clipped_to_zero_on_bad_prediction():
    score = confidence_value(pred_std=200.0, error=500.0)
    assert score == 0.0


def test_confidence_value_close_to_one_on_good_prediction():
    score = confidence_value(pred_std=2.0, error=5.0)
    assert score > 0.9


def test_status_critical_on_urgent_action():
    assert status_from_metrics(0.5, 30.0, "RETRAIN_URGENT") == "Critical"


def test_status_warning_on_retrain_action():
    assert status_from_metrics(0.3, 30.0, "RETRAIN") == "Warning"


def test_status_healthy_on_stable():
    assert status_from_metrics(0.1, 20.0, "STABLE") == "Healthy"


def _make_window(n=10):
    window = deque(maxlen=60)
    for _ in range(n):
        row = {f: 1.0 for f in FEATURE_COLUMNS}
        window.append(row)
    return window


def test_feature_scores_returns_top_n():
    window = _make_window()
    means = {f: 1.0 for f in FEATURE_COLUMNS}
    stds = {f: 1.0 for f in FEATURE_COLUMNS}
    result = feature_scores(window, means, stds, FEATURE_COLUMNS, top_n=5)
    assert len(result) == 5
    assert all("feature" in r and "score" in r for r in result)


def test_feature_scores_empty_window_returns_defaults():
    result = feature_scores(deque(), {}, {})
    assert len(result) > 0


def test_confidence_histogram_buckets():
    confs = deque([0.1, 0.5, 0.9, 0.95])
    hist = confidence_histogram(confs)
    assert len(hist) == 10
    total = sum(b["count"] for b in hist)
    assert total == 4
