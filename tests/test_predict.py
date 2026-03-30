import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ml.predict import predict


def _fit_toy_model():
    X = pd.DataFrame(
        {
            "sensor_1": [0.0, 1.0, 2.0, 3.0],
            "sensor_2": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y = [10.0, 9.0, 8.0, 7.0]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(random_state=42, n_estimators=20)
    model.fit(X_scaled, y)
    return model, scaler


def test_predict_handles_extra_non_feature_keys():
    model, scaler = _fit_toy_model()

    data = {
        "unit": 1,
        "cycle": 1,
        "RUL": 10,
        "sensor_1": 1.5,
        "sensor_2": 2.5,
        "irrelevant": 99,
    }

    pred = predict(model, scaler, data)
    assert isinstance(pred, float)


def test_predict_raises_for_missing_expected_feature():
    model, scaler = _fit_toy_model()

    bad_data = {
        "unit": 1,
        "cycle": 1,
        "RUL": 10,
        "sensor_1": 1.5,
    }

    with pytest.raises(ValueError):
        predict(model, scaler, bad_data)
