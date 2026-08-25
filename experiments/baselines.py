"""Experiment strategy implementations.

Each strategy consumes the same deterministic stream events and scenario
configuration.  Strategies differ only in how they react to drift/retraining
signals.
"""

from __future__ import annotations

from collections import deque
import time

import numpy as np
import pandas as pd

from decision.adaptive_cooldown import AdaptiveCooldown
from decision.engine import DecisionEngine
from drift.adwin_detector import DriftDetector
from drift.anomaly_detector import AnomalyDetector
from drift.data_drift import DataDriftDetector
from drift.error_monitor import ErrorMonitor
from experiments.config import ExperimentConfig
from experiments.data_stream import StreamEvent
from metrics.calculator import FEATURE_COLUMNS
from ml.evaluation import split_training_and_validation
from ml.performance_gate import ModelPerformanceGate
from ml.shadow_evaluator import ShadowModelEvaluator
from ml.train import train_model_with_holdout


def _predict_fast(model, scaler, data: dict) -> float:
    feature_names = getattr(scaler, "feature_names_in_", FEATURE_COLUMNS)
    values = np.array([[float(data.get(feature, 0.0)) for feature in feature_names]])
    scaled = scaler.transform(pd.DataFrame(values, columns=feature_names))
    return float(model.predict(scaled)[0])


class ExperimentStrategy:
    def __init__(
        self,
        config: ExperimentConfig,
        train_df: pd.DataFrame,
        baseline_stds: dict[str, float],
    ):
        self.config = config
        self.train_df = train_df
        self.baseline_stds = baseline_stds
        self.model, self.scaler, _ = train_model_with_holdout(train_df, min_retrain_rows=30)
        self.model_version = 1
        self.expected_columns = train_df.columns.tolist()
        self.buffer = deque(maxlen=240)
        self.error_monitor = ErrorMonitor(window_size=config.error_window)
        self.adwin_detector = DriftDetector()
        self.data_drift_detector = DataDriftDetector(
            window_size=config.data_drift_window,
            p_threshold=config.data_drift_p_threshold,
            drift_feature_ratio_threshold=config.data_drift_feature_ratio_threshold,
            min_effect_size=config.data_drift_min_effect_size,
        )
        self.anomaly_detector = AnomalyDetector(contamination=0.05, random_state=config.seed)
        self.anomaly_detector.fit(train_df)
        self.engine = DecisionEngine(error_threshold=config.error_threshold)
        self.cooldown = AdaptiveCooldown(
            min_cooldown=config.cooldown,
            max_cooldown=config.cooldown,
        )
        self.gate = ModelPerformanceGate(config.performance_gate_threshold)
        self.shadow = ShadowModelEvaluator(
            window_size=config.shadow_window,
            improvement_threshold=config.performance_gate_threshold,
        )
        self.candidate_counter = 0
        self.last_retrain_sample = -10**9
        self.shadow_started_at: int | None = None
        self.shadow_candidate_id: str | None = None
        self.shadow_candidate_validation_mae: float | None = None

    def process(self, event: StreamEvent, run_id: str) -> dict:
        started = time.perf_counter()
        actual = float(event.data["RUL"])
        prediction = _predict_fast(self.model, self.scaler, event.data)
        inference_latency = time.perf_counter() - started
        error = abs(actual - prediction)
        rolling_mae = self.error_monitor.update(error)
        rolling_rmse = self._rolling_rmse()
        trend = self.error_monitor.is_increasing()
        is_anomaly, _ = self.anomaly_detector.is_anomaly(event.data)
        concept_drift = self.adwin_detector.update(rolling_mae if rolling_mae is not None else error)
        drift_details = self.data_drift_detector.update_with_details(event.data)
        feature_drift = bool(drift_details["drift_detected"])
        drift_score = self._drift_score(drift_details, rolling_mae, concept_drift)
        action = self.engine.decide(feature_drift or concept_drift, rolling_mae, trend, drift_score)

        row = self._base_row(
            run_id,
            event,
            actual,
            prediction,
            error,
            rolling_mae,
            rolling_rmse,
            is_anomaly,
            feature_drift,
            concept_drift,
            action,
            drift_score,
            inference_latency,
        )

        if self.shadow.is_evaluating:
            self._update_shadow(event, actual, error, row)

        self.buffer.append(event.data.copy())
        self._maybe_adapt(event, action, drift_score, rolling_mae, row)
        return row

    def _maybe_adapt(self, event, action, drift_score, rolling_mae, row) -> None:
        return None

    def _drift_triggered(self, action: str, drift_score: float, rolling_mae: float | None) -> bool:
        strong_signal = action in {"RETRAIN", "RETRAIN_URGENT"}
        error_signal = rolling_mae is not None and rolling_mae >= self.config.retrain_error_threshold
        return strong_signal and (error_signal or drift_score >= self.config.retrain_drift_score_threshold)

    def _cooldown_ready(self, sample_index: int) -> bool:
        return sample_index - self.last_retrain_sample >= self.config.cooldown

    def _train_candidate(
        self,
    ) -> tuple[object | None, object | None, pd.DataFrame, float, str | None, dict]:
        """Attempt to train a candidate model from the current buffer.

        Returns:
            (model, scaler, validation_df, training_time, skip_reason, validation_log)

            If validation-quality requirements are not met, model and scaler
            are None, validation_df is empty, and skip_reason is a non-None
            string describing why training was skipped.  The caller must record
            this reason without generating a candidate or running the gate.
            
            validation_log contains detailed metrics about buffer composition,
            train/validation split, and unit disjointness verification.
        """
        validation_log = {}
        buffer_df = pd.DataFrame(list(self.buffer))
        
        # Log buffer composition
        validation_log["buffer_rows"] = len(buffer_df)
        validation_log["buffer_units"] = (
            int(buffer_df["unit"].nunique()) if "unit" in buffer_df.columns else 0
        )
        
        if len(buffer_df) < self.config.minimum_retraining_samples:
            validation_log["skip_reason"] = "insufficient_buffer"
            return None, None, pd.DataFrame(), 0.0, "insufficient_buffer", validation_log

        buffer_df = buffer_df[[c for c in self.expected_columns if c in buffer_df.columns]]
        train_df, validation_df = split_training_and_validation(
            buffer_df,
            validation_fraction=self.config.validation_fraction,
            random_state=self.config.seed,
            min_validation_rows=self.config.minimum_validation_rows,
        )

        # Log train/validation split details
        validation_log["train_rows"] = len(train_df)
        validation_log["validation_rows"] = len(validation_df)
        validation_log["train_units"] = (
            int(train_df["unit"].nunique()) if "unit" in train_df.columns else 0
        )
        validation_log["validation_units"] = (
            int(validation_df["unit"].nunique()) if "unit" in validation_df.columns else 0
        )
        
        # Verify unit disjointness
        if "unit" in train_df.columns and "unit" in validation_df.columns:
            train_units = set(train_df["unit"].unique())
            val_units = set(validation_df["unit"].unique())
            intersection = train_units & val_units
            validation_log["unit_disjoint"] = len(intersection) == 0
            validation_log["unit_intersection_count"] = len(intersection)
            if intersection:
                validation_log["unit_intersection"] = sorted(int(u) for u in intersection)
        else:
            validation_log["unit_disjoint"] = None
            validation_log["unit_intersection_count"] = 0
        
        n_val_rows = len(validation_df)
        n_val_units = validation_log["validation_units"]
        
        if n_val_rows < self.config.minimum_validation_rows:
            skip_reason = f"validation_too_small:{n_val_rows}_rows"
            validation_log["skip_reason"] = skip_reason
            return (
                None,
                None,
                pd.DataFrame(),
                0.0,
                skip_reason,
                validation_log,
            )
        if n_val_units < self.config.minimum_validation_units:
            skip_reason = f"validation_too_few_units:{n_val_units}_units"
            validation_log["skip_reason"] = skip_reason
            return (
                None,
                None,
                pd.DataFrame(),
                0.0,
                skip_reason,
                validation_log,
            )

        started = time.perf_counter()
        model, scaler, _ = train_model_with_holdout(
            train_df, min_retrain_rows=30, random_state=self.config.seed
        )
        elapsed = time.perf_counter() - started
        validation_log["training_time"] = elapsed
        return model, scaler, validation_df, elapsed, None, validation_log

    def _replace_model(self, model, scaler) -> None:
        self.model = model
        self.scaler = scaler
        self.model_version += 1

    def _update_shadow(self, event, actual, production_error, row) -> None:
        shadow_pred = _predict_fast(self.shadow.shadow_model, self.shadow.shadow_scaler, event.data)
        self.shadow.production_errors.append(production_error)
        self.shadow.shadow_errors.append(abs(actual - shadow_pred))
        row["shadow_started"] = False
        if len(self.shadow.shadow_errors) < self.shadow.window_size:
            return

        prod_mae = float(np.mean(self.shadow.production_errors))
        shadow_mae = float(np.mean(self.shadow.shadow_errors))
        passed = shadow_mae < prod_mae * self.shadow.improvement_threshold
        row["shadow_completed"] = True
        row["shadow_result"] = "passed" if passed else "failed"
        row["shadow_passed"] = passed
        row["shadow_rejected"] = not passed
        row["production_mae"] = prod_mae
        row["candidate_mae"] = shadow_mae
        row["shadow_evaluation_time"] = float(
            max(0, event.sample_index - (self.shadow_started_at or event.sample_index) + 1)
        )

        if passed:
            self._replace_model(self.shadow.shadow_model, self.shadow.shadow_scaler)
            row["model_promoted"] = True
            row["promotion_decision"] = "promoted"
            row["degraded_promotion"] = (
                self.shadow_candidate_validation_mae is not None
                and shadow_mae > self.shadow_candidate_validation_mae
            )
            row["model_version"] = self.model_version
        else:
            row["promotion_decision"] = "rejected"
        self.shadow.stop_evaluation()
        self.shadow_candidate_id = None
        self.shadow_candidate_validation_mae = None
        self.shadow_started_at = None

    def _base_row(
        self,
        run_id,
        event,
        actual,
        prediction,
        error,
        rolling_mae,
        rolling_rmse,
        is_anomaly,
        feature_drift,
        concept_drift,
        action,
        drift_score,
        inference_latency,
    ) -> dict:
        return {
            "run_id": run_id,
            "seed": self.config.seed,
            "strategy": self.config.strategy,
            "scenario": self.config.scenario,
            "sample_index": event.sample_index,
            "engine_id": event.engine_id,
            "cycle": event.cycle,
            "event_index": event.sample_index,
            "scenario_active": event.scenario_active,
            "degradation_started": event.degradation_started,
            "actual_rul": actual,
            "predicted_rul": prediction,
            "absolute_error": error,
            "squared_error": error**2,
            "rolling_mae": rolling_mae,
            "rolling_rmse": rolling_rmse,
            "anomaly_detected": bool(is_anomaly),
            "feature_drift_detected": bool(feature_drift),
            "concept_drift_detected": bool(concept_drift),
            "drift_score": drift_score,
            "drift_trigger": action in {"RETRAIN", "RETRAIN_URGENT"},
            # --- Granular retraining lifecycle columns ---
            "retraining_triggered": False,
            "retraining_started": False,
            "retraining_completed": False,
            "validation_skipped": False,
            "validation_skip_reason": None,
            "candidate_generated": False,
            "candidate_id": None,
            "candidate_mae": None,
            "candidate_rmse": None,
            "production_mae": None,
            "production_rmse": None,
            "improvement": None,
            "gate_passed": None,
            "gate_rejected": False,
            "shadow_started": False,
            "shadow_completed": False,
            "shadow_passed": False,
            "shadow_rejected": False,
            # shadow_result kept for backward compatibility
            "shadow_result": None,
            "model_promoted": False,
            "promotion_decision": None,
            "degraded_promotion": None,
            "model_version": self.model_version,
            "training_time": 0.0,
            "shadow_evaluation_time": 0.0,
            "inference_latency": inference_latency,
            # --- Validation logging fields (populated during retraining) ---
            "val_buffer_rows": None,
            "val_train_rows": None,
            "val_validation_rows": None,
            "val_buffer_units": None,
            "val_train_units": None,
            "val_validation_units": None,
            "val_unit_disjoint": None,
            "val_unit_intersection_count": None,
            "val_training_time": None,
        }

    def _rolling_rmse(self) -> float | None:
        if len(self.error_monitor.errors) < self.config.error_window:
            return None
        return float(np.mean([error**2 for error in self.error_monitor.errors]) ** 0.5)

    @staticmethod
    def _drift_score(drift_details, rolling_mae, concept_drift) -> float:
        score = float(drift_details["drift_score"])
        if rolling_mae is not None:
            score = max(score, min(rolling_mae / 70.0, 1.0) * 0.65)
        if concept_drift:
            score = max(score, 0.55)
        return float(score)


class StaticStrategy(ExperimentStrategy):
    pass


class ScheduledStrategy(ExperimentStrategy):
    def _maybe_adapt(self, event, action, drift_score, rolling_mae, row) -> None:
        if event.sample_index == 0 or event.sample_index % self.config.retraining_interval != 0:
            return
        self._train_and_replace(row, gated=False)

    def _train_and_replace(self, row, gated: bool) -> None:
        row["retraining_triggered"] = True
        row["retraining_started"] = True
        model, scaler, _, training_time, skip_reason, val_log = self._train_candidate()
        row["training_time"] = training_time
        row["retraining_completed"] = True
        
        # Add validation logging
        for key, value in val_log.items():
            row[f"val_{key}"] = value
        
        if skip_reason is not None:
            row["validation_skipped"] = True
            row["validation_skip_reason"] = skip_reason
            return
        if model is None or scaler is None:
            return
        self.candidate_counter += 1
        row["candidate_generated"] = True
        row["candidate_id"] = f"candidate-{self.candidate_counter}"
        self._replace_model(model, scaler)
        row["model_promoted"] = True
        row["promotion_decision"] = "immediate_replace"


class NaiveAdaptiveStrategy(ScheduledStrategy):
    def _maybe_adapt(self, event, action, drift_score, rolling_mae, row) -> None:
        if event.sample_index < self.config.adaptation_start_index:
            return
        if not self._drift_triggered(action, drift_score, rolling_mae):
            return
        if not self._cooldown_ready(event.sample_index):
            return
        self.last_retrain_sample = event.sample_index
        self._train_and_replace(row, gated=False)


class ProposedStrategy(ExperimentStrategy):
    def _maybe_adapt(self, event, action, drift_score, rolling_mae, row) -> None:
        if event.sample_index < self.config.adaptation_start_index:
            return
        if self.shadow.is_evaluating:
            return
        if not self._drift_triggered(action, drift_score, rolling_mae):
            return
        if not self._cooldown_ready(event.sample_index):
            return

        self.last_retrain_sample = event.sample_index
        row["retraining_triggered"] = True
        row["retraining_started"] = True
        model, scaler, validation_df, training_time, skip_reason, val_log = self._train_candidate()
        row["training_time"] = training_time
        row["retraining_completed"] = True
        
        # Add validation logging
        for key, value in val_log.items():
            row[f"val_{key}"] = value

        if skip_reason is not None:
            row["validation_skipped"] = True
            row["validation_skip_reason"] = skip_reason
            return

        if model is None or scaler is None or validation_df.empty:
            return

        self.candidate_counter += 1
        candidate_id = f"candidate-{self.candidate_counter}"
        row["candidate_generated"] = True
        row["candidate_id"] = candidate_id
        accepted, production_result, candidate_result, improvement, reason = (
            self.gate.should_accept_candidate(
                self.model, self.scaler, model, scaler, validation_df
            )
        )
        row["gate_passed"] = accepted
        row["gate_rejected"] = not accepted
        row["production_mae"] = production_result.mae if production_result else None
        row["production_rmse"] = production_result.rmse if production_result else None
        row["candidate_mae"] = candidate_result.mae if candidate_result else None
        row["candidate_rmse"] = candidate_result.rmse if candidate_result else None
        row["improvement"] = improvement

        if not accepted:
            row["promotion_decision"] = f"gate_rejected:{reason}"
            return

        self.shadow.start_shadow_evaluation(model, scaler)
        self.shadow_started_at = event.sample_index
        self.shadow_candidate_id = candidate_id
        self.shadow_candidate_validation_mae = row["candidate_mae"]
        row["shadow_started"] = True


def make_strategy(config: ExperimentConfig, train_df, baseline_stds) -> ExperimentStrategy:
    if config.strategy == "static":
        return StaticStrategy(config, train_df, baseline_stds)
    if config.strategy == "scheduled":
        return ScheduledStrategy(config, train_df, baseline_stds)
    if config.strategy == "naive_adaptive":
        return NaiveAdaptiveStrategy(config, train_df, baseline_stds)
    if config.strategy == "proposed":
        return ProposedStrategy(config, train_df, baseline_stds)
    raise ValueError(f"Unknown strategy: {config.strategy}")
