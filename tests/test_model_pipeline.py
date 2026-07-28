import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ml.performance_gate import ModelPerformanceGate
from ml.shadow_evaluator import ShadowModelEvaluator


def _fit_toy_model(bias=0.0):
    """Return a model + scaler that predicts (sensor_1 * 3 + bias)."""
    X = pd.DataFrame({"sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    y = [v * 3.0 + bias for v in X["sensor_1"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler


def _make_buffer_df(n=10, bias=0.0):
    rows = [
        {"sensor_1": float(i), "RUL": float(i * 3.0 + bias)}
        for i in range(1, n + 1)
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# ModelPerformanceGate
# ------------------------------------------------------------------

class TestModelPerformanceGate:
    def test_rejects_none_new_model(self):
        gate = ModelPerformanceGate()
        ok, _, _, reason = gate.should_accept_new_model(
            None, None, None, None, None, _make_buffer_df()
        )
        assert ok is False
        assert reason == "new_model_is_none"

    def test_rejects_none_new_mae(self):
        gate = ModelPerformanceGate()
        model, scaler = _fit_toy_model()
        ok, _, _, reason = gate.should_accept_new_model(
            model, scaler, model, scaler, None, _make_buffer_df()
        )
        assert ok is False
        assert reason == "new_mae_is_none"

    def test_accepts_when_no_baseline(self):
        gate = ModelPerformanceGate()
        model, scaler = _fit_toy_model()
        # Buffer too small to evaluate current model → accept new one
        ok, _, _, reason = gate.should_accept_new_model(
            model, scaler, model, scaler, 5.0, _make_buffer_df(n=2)
        )
        assert ok is True
        assert reason == "no_baseline_comparison"

    def test_rejects_degraded_model(self):
        gate = ModelPerformanceGate(improvement_threshold=0.95)
        prod_model, prod_scaler = _fit_toy_model(bias=0.0)
        bad_model, bad_scaler = _fit_toy_model(bias=100.0)
        buf = _make_buffer_df(n=10)
        ok, _, _, reason = gate.should_accept_new_model(
            prod_model, prod_scaler, bad_model, bad_scaler, 999.0, buf
        )
        assert ok is False
        assert "degraded" in reason

    def test_evaluate_model_returns_none_for_small_buffer(self):
        gate = ModelPerformanceGate()
        model, scaler = _fit_toy_model()
        result = gate.evaluate_model_on_buffer(model, scaler, _make_buffer_df(n=3))
        assert result is None


# ------------------------------------------------------------------
# ShadowModelEvaluator
# ------------------------------------------------------------------

class TestShadowModelEvaluator:
    def test_not_evaluating_before_start(self):
        ev = ShadowModelEvaluator(window_size=5)
        assert ev.is_evaluating is False
        assert ev.get_status()["is_evaluating"] is False

    def test_start_sets_evaluating_flag(self):
        ev = ShadowModelEvaluator(window_size=5)
        model, scaler = _fit_toy_model()
        ev.start_shadow_evaluation(model, scaler)
        assert ev.is_evaluating is True

    def test_stop_clears_state(self):
        ev = ShadowModelEvaluator(window_size=5)
        model, scaler = _fit_toy_model()
        ev.start_shadow_evaluation(model, scaler)
        ev.stop_evaluation()
        assert ev.is_evaluating is False
        assert ev.shadow_model is None

    def test_evaluate_both_returns_false_when_not_started(self):
        ev = ShadowModelEvaluator(window_size=3)
        model, scaler = _fit_toy_model()
        result = ev.evaluate_both(model, scaler, {"sensor_1": 2.0}, actual=6.0)
        assert result == (False, None, None)

    def test_promotes_when_shadow_is_better(self):
        ev = ShadowModelEvaluator(window_size=5, improvement_threshold=0.95)
        prod_model, prod_scaler = _fit_toy_model(bias=50.0)   # bad production
        shadow_model, shadow_scaler = _fit_toy_model(bias=0.0) # good shadow
        ev.start_shadow_evaluation(shadow_model, shadow_scaler)

        should_promote = False
        for i in range(1, 10):
            data = {"sensor_1": float(i)}
            actual = float(i * 3.0)
            should_promote, _, _ = ev.evaluate_both(prod_model, prod_scaler, data, actual)
            if should_promote:
                break

        assert should_promote == True  # noqa: E712 — np.bool_ safe comparison

    def test_get_status_tracks_sample_count(self):
        ev = ShadowModelEvaluator(window_size=5)
        model, scaler = _fit_toy_model()
        ev.start_shadow_evaluation(model, scaler)
        for i in range(3):
            ev.evaluate_both(model, scaler, {"sensor_1": float(i + 1)}, actual=float((i + 1) * 3))
        status = ev.get_status()
        assert status["samples_collected"] == 3
        assert status["ready_to_decide"] is False
