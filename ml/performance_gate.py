from __future__ import annotations

from ml.evaluation import EvaluationResult, evaluate_regressor


class ModelPerformanceGate:
    """
    Ensures new models are better than current production model before deployment.
    Prevents model degradation by comparing both models on the exact same
    validation examples and target values.
    """
    
    def __init__(self, improvement_threshold=0.95):
        """
        Args:
            improvement_threshold: Backward-compatible threshold. The existing
                value 0.95 means candidate MAE must be at least 5% lower than
                production MAE.
        """
        self.improvement_threshold = improvement_threshold

    @property
    def required_improvement_fraction(self):
        """Return the minimum fractional MAE improvement required to pass."""

        if self.improvement_threshold >= 0.5:
            return max(0.0, 1.0 - float(self.improvement_threshold))
        return max(0.0, float(self.improvement_threshold))
    
    def evaluate_model_on_buffer(self, model, scaler, buffer_df):
        """Evaluate a model on buffered data and return MAE for legacy callers."""

        if buffer_df is None or len(buffer_df) < 5:
            return None

        try:
            result = evaluate_regressor(model, scaler, buffer_df)
        except Exception:
            return None

        if result is None or result.n_samples < 5:
            return None
        return result.mae

    def evaluate_model(self, model, scaler, validation_df):
        """Evaluate a model and return MAE/RMSE on the supplied validation frame."""

        if validation_df is None or len(validation_df) < 5:
            return None
        try:
            result = evaluate_regressor(model, scaler, validation_df)
        except Exception:
            return None
        if result is None or result.n_samples < 5:
            return None
        return result

    def should_accept_candidate(
        self,
        current_model,
        current_scaler,
        new_model,
        new_scaler,
        common_validation_df,
    ):
        """
        Compare production and candidate on one common validation dataset.

        Returns:
            (should_accept, production_result, candidate_result, improvement, reason)
        """

        if new_model is None or new_scaler is None:
            return False, None, None, None, "new_model_is_none"

        production_result = self.evaluate_model(
            current_model, current_scaler, common_validation_df
        )
        candidate_result = self.evaluate_model(new_model, new_scaler, common_validation_df)

        if production_result is None:
            return False, None, candidate_result, None, "production_validation_failed"
        if candidate_result is None:
            return False, production_result, None, None, "candidate_validation_failed"

        epsilon = 1e-12
        if production_result.mae <= epsilon:
            if candidate_result.mae <= epsilon:
                return False, production_result, candidate_result, 0.0, "no_measurable_improvement"
            return False, production_result, candidate_result, None, "production_mae_near_zero"

        improvement = (production_result.mae - candidate_result.mae) / production_result.mae
        required = self.required_improvement_fraction
        if improvement >= required:
            return (
                True,
                production_result,
                candidate_result,
                float(improvement),
                f"improved_by_{improvement * 100:.1f}%",
            )

        return (
            False,
            production_result,
            candidate_result,
            float(improvement),
            f"insufficient_improvement_{improvement * 100:.1f}%",
        )
    
    def should_accept_new_model(self, current_model, current_scaler, 
                                 new_model, new_scaler, new_mae, 
                                 validation_buffer):
        """
        Compare current model vs new model on same validation data.
        
        Returns:
            (should_accept: bool, current_mae: float, new_mae: float, reason: str)
        """
        accepted, production_result, candidate_result, _, reason = (
            self.should_accept_candidate(
                current_model,
                current_scaler,
                new_model,
                new_scaler,
                validation_buffer,
            )
        )

        current_mae = production_result.mae if production_result is not None else None
        candidate_mae = candidate_result.mae if candidate_result is not None else None
        return accepted, current_mae, candidate_mae, reason
