import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ml.confidence_predictor import ConfidencePredictor


def _fit_toy_model():
    X = pd.DataFrame({"sensor_1": [0.0, 1.0, 2.0, 3.0], "sensor_2": [1.0, 2.0, 3.0, 4.0]})
    y = [10.0, 9.0, 8.0, 7.0]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(random_state=42, n_estimators=20)
    model.fit(X_scaled, y)
    return model, scaler


def test_predict_with_confidence_returns_four_floats():
    model, scaler = _fit_toy_model()
    cp = ConfidencePredictor(confidence_level=0.9)
    data = {"sensor_1": 1.5, "sensor_2": 2.5}
    pred, lower, upper, std = cp.predict_with_confidence(model, scaler, data)
    assert isinstance(pred, float)
    assert lower <= pred <= upper
    assert std >= 0.0


def test_confidence_interval_contains_prediction():
    model, scaler = _fit_toy_model()
    cp = ConfidencePredictor()
    data = {"sensor_1": 1.0, "sensor_2": 2.0}
    pred, lower, upper, _ = cp.predict_with_confidence(model, scaler, data)
    assert lower <= pred <= upper


def test_get_confidence_category_boundaries():
    cp = ConfidencePredictor()
    assert cp.get_confidence_category(3.0) == "high_confidence"
    assert cp.get_confidence_category(7.0) == "medium_confidence"
    assert cp.get_confidence_category(15.0) == "low_confidence"


def test_predict_raises_on_missing_feature():
    model, scaler = _fit_toy_model()
    cp = ConfidencePredictor()
    with pytest.raises(ValueError):
        cp.predict_with_confidence(model, scaler, {"sensor_1": 1.0})
