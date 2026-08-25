"""Model scoring helpers used by experiment strategies."""

from __future__ import annotations

from ml.evaluation import EvaluationResult, evaluate_regressor


def score_model(model, scaler, validation_df) -> EvaluationResult | None:
    return evaluate_regressor(model, scaler, validation_df)
