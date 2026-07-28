import pandas as pd
import numpy as np
from ml.train import train_model_with_holdout


def _make_df(n_units=5, cycles_per_unit=30):
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            row = {"unit": unit, "cycle": cycle}
            for i in range(1, 4):
                row[f"op_setting_{i}"] = float(np.random.rand())
            for i in range(1, 22):
                row[f"sensor_{i}"] = float(np.random.rand())
            row["RUL"] = float(cycles_per_unit - cycle)
            rows.append(row)
    return pd.DataFrame(rows)


def test_returns_model_scaler_and_mae():
    df = _make_df()
    model, scaler, mae = train_model_with_holdout(df, min_retrain_rows=30)
    assert model is not None
    assert scaler is not None
    assert isinstance(mae, float)
    assert mae >= 0.0


def test_returns_none_tuple_when_too_few_rows():
    tiny_df = _make_df(n_units=1, cycles_per_unit=5)
    model, scaler, mae = train_model_with_holdout(tiny_df, min_retrain_rows=100)
    assert model is None
    assert scaler is None
    assert mae is None


def test_single_unit_fallback_uses_cycle_split():
    df = _make_df(n_units=1, cycles_per_unit=40)
    model, scaler, mae = train_model_with_holdout(df, min_retrain_rows=10)
    assert model is not None


def test_model_produces_predictions():
    df = _make_df()
    model, scaler, _ = train_model_with_holdout(df, min_retrain_rows=30)
    X = df.drop(columns=["RUL", "unit", "cycle"]).iloc[:5]
    preds = model.predict(scaler.transform(X))
    assert len(preds) == 5
