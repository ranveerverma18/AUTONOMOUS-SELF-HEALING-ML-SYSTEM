import numpy as np
import pandas as pd

from ml.evaluation import evaluate_regressor, split_training_and_validation
from ml.performance_gate import ModelPerformanceGate


class PassthroughScaler:
    feature_names_in_ = np.array(["sensor_1"])

    def transform(self, X):
        return X.to_numpy(dtype=float)


class RecordingModel:
    def __init__(self, offset=0.0):
        self.offset = offset
        self.seen = []

    def predict(self, X):
        self.seen.append(X.copy())
        return X[:, 0] + self.offset


def _validation_df():
    return pd.DataFrame(
        {
            "unit": [1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5],
            "sensor_1": [10.0, 20.0, 30.0, 40.0, 50.0],
            "RUL": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


def test_both_models_receive_identical_validation_examples():
    gate = ModelPerformanceGate(improvement_threshold=0.95)
    prod = RecordingModel(offset=10.0)
    candidate = RecordingModel(offset=0.0)
    scaler = PassthroughScaler()

    accepted, _, _, improvement, _ = gate.should_accept_candidate(
        prod, scaler, candidate, scaler, _validation_df()
    )

    assert accepted is True
    assert improvement == 1.0
    assert np.array_equal(prod.seen[0], candidate.seen[0])


def test_mae_and_rmse_calculation_is_correct():
    result = evaluate_regressor(RecordingModel(offset=2.0), PassthroughScaler(), _validation_df())

    assert result.mae == 2.0
    assert result.rmse == 2.0
    assert result.n_samples == 5


def test_candidate_without_sufficient_improvement_fails():
    gate = ModelPerformanceGate(improvement_threshold=0.95)
    scaler = PassthroughScaler()

    accepted, _, _, improvement, _ = gate.should_accept_candidate(
        RecordingModel(offset=10.0),
        scaler,
        RecordingModel(offset=9.6),
        scaler,
        _validation_df(),
    )

    assert accepted is False
    assert improvement < 0.05


def test_near_zero_production_mae_is_handled_safely():
    gate = ModelPerformanceGate(improvement_threshold=0.95)
    scaler = PassthroughScaler()

    accepted, _, _, improvement, reason = gate.should_accept_candidate(
        RecordingModel(offset=0.0),
        scaler,
        RecordingModel(offset=0.0),
        scaler,
        _validation_df(),
    )

    assert accepted is False
    assert improvement == 0.0
    assert reason == "no_measurable_improvement"


def test_validation_split_is_reproducible_and_separate():
    df = pd.concat([_validation_df().assign(unit=unit) for unit in range(1, 5)])

    train_a, val_a = split_training_and_validation(df, random_state=7)
    train_b, val_b = split_training_and_validation(df, random_state=7)

    assert val_a.equals(val_b)
    assert train_a.equals(train_b)
    assert set(train_a["unit"]).isdisjoint(set(val_a["unit"]))
