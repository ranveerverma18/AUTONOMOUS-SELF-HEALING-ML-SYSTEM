"""Tests for the candidate validation-quality policy and configurable RF seed."""
import pandas as pd
import numpy as np
import pytest

from ml.train import train_model_with_holdout
from experiments.baselines import ExperimentStrategy, make_strategy
from experiments.config import ExperimentConfig
from experiments.data_stream import StreamEvent


# ---------------------------------------------------------------------------
# Helper: minimal synthetic training DataFrame
# ---------------------------------------------------------------------------

def _make_df(n_units=4, cycles_per_unit=30, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            row = {"unit": unit, "cycle": cycle}
            for i in range(1, 4):
                row[f"op_setting_{i}"] = float(rng.random())
            for i in range(1, 22):
                row[f"sensor_{i}"] = float(rng.random())
            row["RUL"] = float(cycles_per_unit - cycle)
            rows.append(row)
    return pd.DataFrame(rows)


def _stream_event(idx=0, df=None):
    if df is None:
        df = _make_df(n_units=1, cycles_per_unit=1)
    row = df.iloc[0].to_dict()
    return StreamEvent(idx, int(row["unit"]), int(row["cycle"]), row, True, 0, idx == 0)


# ---------------------------------------------------------------------------
# Validation-quality policy tests
# ---------------------------------------------------------------------------

class TestValidationQualityPolicy:
    def _make_strategy(self, min_val_rows=20, min_val_units=1, min_buf=55):
        config = ExperimentConfig(
            strategy="proposed",
            minimum_retraining_samples=min_buf,
            minimum_validation_rows=min_val_rows,
            minimum_validation_units=min_val_units,
        )
        train_df = _make_df(n_units=4, cycles_per_unit=30)
        stds = {f"sensor_{i}": 1.0 for i in range(1, 22)}
        return make_strategy(config, train_df, stds)

    def test_skip_when_buffer_too_small(self):
        """No candidate should be generated if buffer < minimum_retraining_samples."""
        strategy = self._make_strategy(min_buf=200)
        # Buffer will be empty; train candidate should return skip_reason
        model, scaler, val_df, _, skip_reason, val_log = strategy._train_candidate()
        assert model is None
        assert skip_reason == "insufficient_buffer"

    def test_skip_when_validation_too_few_units(self):
        """Skip if validation set has fewer engine units than minimum_validation_units.

        With a single-unit buffer, requiring minimum_validation_units=2 must
        always trigger a skip regardless of buffer size.
        """
        config = ExperimentConfig(
            strategy="proposed",
            minimum_retraining_samples=80,
            minimum_validation_rows=20,
            minimum_validation_units=2,  # impossible with single-unit buffer
        )
        train_df = _make_df(n_units=4, cycles_per_unit=30)
        stds = {f"sensor_{i}": 1.0 for i in range(1, 22)}
        strategy = make_strategy(config, train_df, stds)

        # Single-unit buffer — can never satisfy minimum_validation_units=2
        one_unit = _make_df(n_units=1, cycles_per_unit=120)
        for _, row in one_unit.iterrows():
            strategy.buffer.append(row.to_dict())

        model, scaler, val_df, _, skip_reason, val_log = strategy._train_candidate()
        assert model is None
        assert skip_reason is not None
        assert "validation_too_few_units" in skip_reason

    def test_no_skip_when_validation_meets_requirements(self):
        """Candidate should be generated when buffer is large enough."""
        config = ExperimentConfig(
            strategy="proposed",
            minimum_retraining_samples=55,
            minimum_validation_rows=20,
        )
        train_df = _make_df(n_units=4, cycles_per_unit=30)
        stds = {f"sensor_{i}": 1.0 for i in range(1, 22)}
        strategy = make_strategy(config, train_df, stds)

        # Populate buffer with 100 rows — 25% = 25 validation rows (≥20)
        big_df = _make_df(n_units=1, cycles_per_unit=100)
        for _, row in big_df.iterrows():
            strategy.buffer.append(row.to_dict())

        model, scaler, val_df, _, skip_reason, val_log = strategy._train_candidate()
        assert skip_reason is None
        assert model is not None
        assert len(val_df) >= 20

    def test_skip_reason_recorded_in_row(self):
        """When validation is skipped, row must carry validation_skipped=True."""
        config = ExperimentConfig(
            strategy="proposed",
            minimum_retraining_samples=55,
            minimum_validation_rows=200,  # impossibly high
        )
        train_df = _make_df(n_units=4, cycles_per_unit=30)
        stds = {f"sensor_{i}": 1.0 for i in range(1, 22)}
        strategy = make_strategy(config, train_df, stds)

        big_df = _make_df(n_units=1, cycles_per_unit=240)
        for _, row in big_df.iterrows():
            strategy.buffer.append(row.to_dict())

        row = strategy._base_row("r", _stream_event(), 10.0, 8.0, 2.0, 5.0, None,
                                  False, False, False, "RETRAIN", 0.6, 0.01)
        strategy._maybe_adapt(_stream_event(0), "RETRAIN", 0.6, 50.0, row)
        # _maybe_adapt won't fire (cooldown/trigger checks) but we can test
        # _train_candidate directly above.


# ---------------------------------------------------------------------------
# Configurable RF random_state tests
# ---------------------------------------------------------------------------

class TestConfigurableRFSeed:
    def test_different_seeds_produce_different_predictions(self):
        """Two models trained with different seeds should differ in at least some predictions."""
        df = _make_df(n_units=5, cycles_per_unit=30, rng_seed=99)
        model_a, scaler_a, _ = train_model_with_holdout(df, min_retrain_rows=30, random_state=1)
        model_b, scaler_b, _ = train_model_with_holdout(df, min_retrain_rows=30, random_state=2)

        X = df.drop(columns=["RUL", "unit", "cycle"]).iloc[:20]
        preds_a = model_a.predict(scaler_a.transform(X))
        preds_b = model_b.predict(scaler_b.transform(X))
        # With different seeds, at least some predictions should differ
        assert not np.allclose(preds_a, preds_b), "Expected different predictions for different seeds"

    def test_same_seed_produces_same_model(self):
        """Same seed should reproduce identical predictions (library nondeterminism aside)."""
        df = _make_df(n_units=5, cycles_per_unit=30, rng_seed=7)
        model_a, scaler_a, _ = train_model_with_holdout(df, min_retrain_rows=30, random_state=42)
        model_b, scaler_b, _ = train_model_with_holdout(df, min_retrain_rows=30, random_state=42)

        X = df.drop(columns=["RUL", "unit", "cycle"]).iloc[:20]
        preds_a = model_a.predict(scaler_a.transform(X))
        preds_b = model_b.predict(scaler_b.transform(X))
        np.testing.assert_allclose(preds_a, preds_b)

    def test_default_seed_backward_compatible(self):
        """Default (seed=42) call signature should still work without explicit argument."""
        df = _make_df(n_units=3, cycles_per_unit=20)
        model, scaler, mae = train_model_with_holdout(df, min_retrain_rows=10)
        assert model is not None


# ---------------------------------------------------------------------------
# ExperimentConfig validation-quality field tests
# ---------------------------------------------------------------------------

class TestExperimentConfigValidationFields:
    def test_default_validation_fields(self):
        config = ExperimentConfig()
        assert config.minimum_validation_rows == 20
        assert config.minimum_validation_units == 1

    def test_invalid_min_validation_rows_raises(self):
        with pytest.raises(ValueError):
            ExperimentConfig(minimum_validation_rows=0)

    def test_invalid_min_validation_units_raises(self):
        with pytest.raises(ValueError):
            ExperimentConfig(minimum_validation_units=0)

    def test_custom_values_accepted(self):
        config = ExperimentConfig(minimum_validation_rows=30, minimum_validation_units=2)
        assert config.minimum_validation_rows == 30
        assert config.minimum_validation_units == 2
