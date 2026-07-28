import pandas as pd
from drift.anomaly_detector import AnomalyDetector


def _make_train_df(n=50):
    return pd.DataFrame(
        {
            "sensor_1": [float(i % 10) for i in range(n)],
            "sensor_2": [float(i % 5) for i in range(n)],
            "RUL": [100 - i for i in range(n)],
            "unit": [1] * n,
            "cycle": list(range(n)),
        }
    )


def test_fit_returns_true_on_enough_data():
    detector = AnomalyDetector()
    df = _make_train_df(50)
    assert detector.fit(df) is True
    assert detector.is_fitted is True


def test_fit_returns_false_on_insufficient_data():
    detector = AnomalyDetector()
    df = _make_train_df(5)
    assert detector.fit(df) is False
    assert detector.is_fitted is False


def test_is_anomaly_returns_bool_and_float():
    detector = AnomalyDetector()
    detector.fit(_make_train_df(50))
    is_anom, score = detector.is_anomaly({"sensor_1": 5.0, "sensor_2": 2.5})
    assert isinstance(is_anom, bool)
    assert isinstance(score, float)


def test_is_anomaly_returns_false_before_fit():
    detector = AnomalyDetector()
    is_anom, score = detector.is_anomaly({"sensor_1": 5.0})
    assert is_anom is False
    assert score == 0.0


def test_outlier_has_lower_score_than_inlier():
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(_make_train_df(100))
    _, inlier_score = detector.is_anomaly({"sensor_1": 5.0, "sensor_2": 2.5})
    _, outlier_score = detector.is_anomaly({"sensor_1": 99999.0, "sensor_2": 99999.0})
    assert outlier_score < inlier_score
