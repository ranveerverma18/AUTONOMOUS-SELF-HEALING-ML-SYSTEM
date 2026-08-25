import numpy as np

from drift.data_drift import DataDriftDetector
from scenarios.correlated_drift import CorrelatedDrift
from scenarios.drift_recovery import DriftRecovery
from scenarios.gradual_drift import GradualDrift
from scenarios.sensor_failure import SensorFailure
from scenarios.sudden_spike import SuddenSpike


FEATURES = [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
BASELINE_STDS = {feature: 1.0 for feature in FEATURES}


def _base_data():
    data = {feature: 5.0 for feature in FEATURES}
    data.update({"unit": 1, "cycle": 1, "RUL": 100.0})
    return data


def _changed_feature_count(scenario_cls, cycle_index=5):
    data = _base_data()
    original = dict(data)
    scenario_cls.apply(data, cycle_index, BASELINE_STDS, np.random.default_rng(7))
    return sum(1 for feature in FEATURES if data[feature] != original[feature])


def _detected_with_dashboard_like_settings(scenario_cls, duration=45):
    detector = DataDriftDetector(
        window_size=30,
        p_threshold=0.01,
        drift_feature_ratio_threshold=0.12,
        min_effect_size=0.10,
    )
    for _ in range(30):
        detector.update_with_details(_base_data())

    result = None
    rng = np.random.default_rng(7)
    for cycle_index in range(duration):
        data = _base_data()
        scenario_cls.apply(data, cycle_index, BASELINE_STDS, rng)
        result = detector.update_with_details(data)
        if result["drift_detected"]:
            return True
    return bool(result and result["drift_detected"])


def test_critical_sudden_spike_is_detectable_within_duration():
    assert _changed_feature_count(SuddenSpike) == 21
    assert _detected_with_dashboard_like_settings(SuddenSpike, duration=45) is True


def test_critical_drift_recovery_is_detectable_before_recovery_phase_ends():
    assert _changed_feature_count(DriftRecovery, cycle_index=5) == 21
    assert _detected_with_dashboard_like_settings(DriftRecovery, duration=60) is True


def test_multi_sensor_drift_scenarios_exceed_feature_ratio_threshold():
    assert _changed_feature_count(GradualDrift, cycle_index=40) == 4
    assert _changed_feature_count(CorrelatedDrift) == 6
    assert _detected_with_dashboard_like_settings(GradualDrift, duration=100) is True
    assert _detected_with_dashboard_like_settings(CorrelatedDrift, duration=60) is True


def test_sensor_failure_is_anomaly_first_not_fleet_level_feature_drift():
    assert _changed_feature_count(SensorFailure) == 2
    assert _detected_with_dashboard_like_settings(SensorFailure, duration=80) is False


def test_drift_recovery_reaches_baseline_at_final_active_cycle():
    data = _base_data()
    DriftRecovery.apply(data, cycle_index=59, baseline_stds=BASELINE_STDS, rng=np.random.default_rng(7))
    assert all(data[f"sensor_{i}"] == 5.0 for i in range(1, 22))
