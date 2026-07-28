"""
api_server.py - Autonomous Self-Healing ML Dashboard Backend

Full pipeline:
  - Random Forest (sklearn) trained via ml/train.py
  - ADWIN concept drift detection (river) via drift/adwin_detector.py
  - KS-test feature drift detection via drift/data_drift.py
  - Isolation Forest anomaly detection via drift/anomaly_detector.py
  - Rolling error monitor via drift/error_monitor.py
  - Confidence intervals from RF tree variance via ml/confidence_predictor.py
  - Performance gate before shadow evaluation via ml/performance_gate.py
  - Shadow A/B evaluation before model promotion via ml/shadow_evaluator.py
  - Adaptive cooldown via decision/adaptive_cooldown.py
  - Decision engine via decision/engine.py
  - Adaptive ingestion rate control via rate_limiting/controller.py
  - Dashboard metric helpers via metrics/calculator.py
  - 8 controlled fault scenarios via scenarios/
"""

from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dataset.processed.preprocess_module import load_data, add_rul
from decision.adaptive_cooldown import AdaptiveCooldown
from decision.engine import DecisionEngine
from drift.adwin_detector import DriftDetector
from drift.anomaly_detector import AnomalyDetector
from drift.data_drift import DataDriftDetector
from drift.error_monitor import ErrorMonitor
from governance.audit_log import AuditLog
from metrics.calculator import (
    confidence_value,
    status_from_metrics,
    feature_scores,
    distribution_shift,
    confidence_histogram,
)
from ml.confidence_predictor import ConfidencePredictor
from ml.performance_gate import ModelPerformanceGate
from ml.shadow_evaluator import ShadowModelEvaluator
from ml.train import train_model_with_holdout
from rate_limiting.controller import RateLimitController, STREAM_QUEUE_MAXLEN
from scenarios.registry import SCENARIO_REGISTRY, get_scenario_list


FEATURE_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)] + [
    f"sensor_{i}" for i in range(1, 22)
]

STREAM_TICK_SECONDS = 0.05
DASHBOARD_SNAPSHOT_SECONDS = 0.5
INITIAL_STREAM_WARMUP_SAMPLES = 24
MAX_EVENTS_PER_WORKER_TICK = 8
MAX_PROCESSING_BURST_EVENTS = 24.0


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnomalyRequest(BaseModel):
    scenario: str = Field(default="sudden_spike")


class ControlRequest(BaseModel):
    simulatedRate: float | None = Field(default=None, ge=0.5, le=30.0)
    rateLimit: float | None = Field(default=None, ge=1.0, le=40.0)
    rateLimitEnabled: bool | None = None


# ---------------------------------------------------------------------------
# Streaming ML Runtime
# ---------------------------------------------------------------------------

class StreamingMLRuntime:
    def __init__(self):
        self.lock = Lock()
        self.rng = np.random.default_rng(42)
        self.base_path = Path(__file__).resolve().parent

        # Stream controls
        self.stream_rate = 8.0
        self.rate_limiter = RateLimitController(rate_limit=14.0, worker_capacity_limit=40.0)
        self.arrival_carry = 0.0
        self.processing_carry = 0.0
        self.current_incoming_rate = self.stream_rate
        self.current_processed_rate = 0.0
        self.last_prediction_std = 5.0
        self._last_controls_audit: tuple = (None, None, None)  # (stream_rate, rate_limit, rate_limit_enabled) at last log
        self.incoming_events_since_snapshot = 0
        self.processed_events_since_snapshot = 0
        self.last_snapshot_at = time.monotonic()
        self.latest_point = None
        self.dashboard_snapshot = None
        self.snapshot_lock = Lock()
        self.stop_event = Event()
        self.worker_thread = None
        self.retrain_job_active = False
        self.retrain_thread = None

        # Model state
        self.model = None
        self.scaler = None
        self.model_version = 1
        self.expected_train_columns: list = []

        # Counters
        self.sample_index = 0
        self.pointer = 0
        self.last_advance = time.monotonic()
        self.last_action = "System online - monitoring active sensor stream"
        self.system_state = "Monitoring"

        # Active scenario state
        self.active_scenario_id = None
        self.active_scenario_remaining = 0
        self.active_scenario_cycle = 0

        # Ring buffers
        self.series = deque(maxlen=80)
        self.buffer = deque(maxlen=240)
        self.confidences = deque(maxlen=120)
        self.current_feature_window = deque(maxlen=60)
        self.action_reasons = Counter()

        # Governance - audit trail, alerts, timeline, model history
        self.audit = AuditLog()

        # Pipeline components
        self.anomaly_detector = AnomalyDetector(contamination=0.05)
        self.adwin_detector = DriftDetector()
        self.data_drift_detector = DataDriftDetector(
            window_size=80,
            p_threshold=0.01,
            drift_feature_ratio_threshold=0.45,
            min_effect_size=0.12,
        )
        self.error_monitor = ErrorMonitor(window_size=8)
        self.confidence_predictor = ConfidencePredictor(confidence_level=0.9)
        self.performance_gate = ModelPerformanceGate(improvement_threshold=0.95)
        self.shadow_evaluator = ShadowModelEvaluator(window_size=20)
        self.engine = DecisionEngine(error_threshold=75)
        self.cooldown = AdaptiveCooldown(min_cooldown=30, max_cooldown=75)

        # Load data and train initial model
        self._load_and_train()

        # Populate initial audit trail
        self.audit.append_timeline("Monitoring system online", "Healthy")
        self.audit.append_model_history(self.model_version)
        self.audit.append_audit(
            "System Initialized",
            "Predictive maintenance model loaded - baseline established from historical engine-unit records",
            f"Active model: v1.0.{self.model_version} | Status: Nominal",
            "Healthy",
            "MODEL",
        )

        self._warm_up_stream()
        self._publish_dashboard_snapshot()
        self.start()

    # -----------------------------------------------------------------------
    # Data loading and initial training
    # -----------------------------------------------------------------------

    def _load_and_train(self):
        raw_path = self.base_path / "dataset" / "raw" / "train_FD001.txt"
        df = load_data(str(raw_path))
        df = add_rul(df)
        df["unit"] = df["unit"].astype(int)
        df["cycle"] = df["cycle"].astype(int)

        # Split units 76/24 train/stream
        units = df["unit"].unique().tolist()
        shuffled = self.rng.permutation(units).tolist()
        split_at = int(len(shuffled) * 0.76)
        train_units = set(shuffled[:split_at])
        stream_units = set(shuffled[split_at:])

        train_df = df[df["unit"].isin(train_units)].reset_index(drop=True)
        stream_df = df[df["unit"].isin(stream_units)].reset_index(drop=True)
        self.expected_train_columns = train_df.columns.tolist()

        self.model, self.scaler, _ = train_model_with_holdout(
            train_df, min_retrain_rows=30
        )

        # Stream records (shuffled)
        stream_records = stream_df.to_dict(orient="records")
        perm = self.rng.permutation(len(stream_records)).tolist()
        self.stream_records = [stream_records[i] for i in perm]

        # Baseline statistics for drift display and scenario scaling
        self.baseline_means = {f: float(train_df[f].mean()) for f in FEATURE_COLUMNS}
        self.baseline_stds = {
            f: float(max(train_df[f].std(), 0.5)) for f in FEATURE_COLUMNS
        }

        # Fit Isolation Forest on training data
        self.anomaly_detector.fit(train_df)

    def _scaled_feature_array(self, data):
        feature_names = getattr(self.scaler, "feature_names_in_", FEATURE_COLUMNS)
        values = np.array([[float(data.get(feature, 0.0)) for feature in feature_names]])
        return (values - self.scaler.mean_) / self.scaler.scale_

    def _predict_with_confidence_fast(self, data):
        scaled = self._scaled_feature_array(data)
        if hasattr(self.model, "estimators_"):
            tree_predictions = np.array(
                [tree.predict(scaled)[0] for tree in self.model.estimators_]
            )
            pred = float(np.mean(tree_predictions))
            pred_std = float(np.std(tree_predictions))
        else:
            pred = float(self.model.predict(scaled)[0])
            pred_std = self.last_prediction_std

        margin = 1.645 * pred_std
        return pred, pred - margin, pred + margin, pred_std

    def _predict_batch_with_confidence(self, events):
        if not events:
            return []

        feature_names = getattr(self.scaler, "feature_names_in_", FEATURE_COLUMNS)
        values = np.array(
            [
                [float(event["data"].get(feature, 0.0)) for feature in feature_names]
                for event in events
            ]
        )
        scaled = (values - self.scaler.mean_) / self.scaler.scale_

        if hasattr(self.model, "estimators_"):
            tree_predictions = np.vstack(
                [tree.predict(scaled) for tree in self.model.estimators_]
            )
            preds = np.mean(tree_predictions, axis=0)
            stds = np.std(tree_predictions, axis=0)
        else:
            preds = self.model.predict(scaled)
            stds = np.full(len(events), self.last_prediction_std)

        margins = 1.645 * stds
        return [
            (
                float(pred),
                float(pred - margin),
                float(pred + margin),
                float(std),
            )
            for pred, margin, std in zip(preds, margins, stds)
        ]

    def _detect_anomalies_batch(self, events):
        if not events:
            return []

        defaults = [(False, 0.0)] * len(events)
        detector = self.anomaly_detector
        if not detector.is_fitted or detector.detector is None:
            return defaults

        try:
            feature_names = detector.feature_names or FEATURE_COLUMNS
            rows = [
                {feature: event["data"].get(feature, 0.0) for feature in feature_names}
                for event in events
            ]
            features_df = pd.DataFrame(rows, columns=feature_names)
            predictions = detector.detector.predict(features_df)
            scores = detector.detector.score_samples(features_df)
            return [
                (bool(prediction == -1), float(score))
                for prediction, score in zip(predictions, scores)
            ]
        except Exception:
            return defaults

    def _predict_model_fast(self, model, scaler, data):
        feature_names = getattr(scaler, "feature_names_in_", FEATURE_COLUMNS)
        values = np.array([[float(data.get(feature, 0.0)) for feature in feature_names]])
        scaled = (values - scaler.mean_) / scaler.scale_
        if hasattr(model, "estimators_"):
            tree_predictions = np.array([tree.predict(scaled)[0] for tree in model.estimators_])
            return float(np.mean(tree_predictions))
        return float(model.predict(scaled)[0])

    # -----------------------------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------------------------

    def _now_parts(self):
        """Convenience timestamp helper (used for latency/point timestamps)."""
        from datetime import datetime as _dt
        now = _dt.now()
        return now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")

    # -----------------------------------------------------------------------
    # Stream helpers
    # -----------------------------------------------------------------------

    def _next_row(self):
        row = dict(self.stream_records[self.pointer % len(self.stream_records)])
        self.pointer += 1
        return row

    def _apply_stream_noise(self, row):
        data = dict(row)
        for feature in FEATURE_COLUMNS:
            scale = self.baseline_stds.get(feature, 1.0)
            data[feature] = float(data[feature] + self.rng.normal(0, 0.015 * scale))
        return data

    def _apply_active_scenario(self, data):
        if self.active_scenario_id is None or self.active_scenario_remaining <= 0:
            return False
        scenario_cls = SCENARIO_REGISTRY.get(self.active_scenario_id)
        if scenario_cls is None:
            return False

        scenario_cls.apply(data, self.active_scenario_cycle, self.baseline_stds, self.rng)
        self.active_scenario_cycle += 1
        self.active_scenario_remaining -= 1

        if self.active_scenario_remaining <= 0:
            self.audit.append_timeline("Sensor readings returned to nominal range", "Healthy")
            self.audit.append_audit(
                "Disturbance Cleared",
                "Anomalous sensor pattern has subsided - all monitored channels are within expected bounds",
                "Drift monitors returning to standard alert thresholds",
                "Healthy",
                "DRIFT",
            )
            self.active_scenario_id = None
            self.active_scenario_cycle = 0

        return True

    # -----------------------------------------------------------------------
    # Metrics helpers  (delegated to metrics/calculator.py)
    # -----------------------------------------------------------------------

    def _feature_scores(self):
        return feature_scores(
            self.current_feature_window,
            self.baseline_means,
            self.baseline_stds,
            FEATURE_COLUMNS,
        )

    def _distribution_shift(self, feature_scores_list):
        return distribution_shift(
            feature_scores_list,
            self.current_feature_window,
            self.baseline_means,
            self.baseline_stds,
        )

    def _histogram(self):
        return confidence_histogram(self.confidences)

    def _update_rate_controller(self, elapsed):
        """Delegate to rate_limiting/controller.py RateLimitController."""
        latest = self.series[-1] if self.series else {}
        latest_drift = float(latest.get("driftScore", 0.0))
        is_retraining = self.retrain_job_active or self.shadow_evaluator.is_evaluating

        # Sync the controller's cached backlog counter from the live queue length.
        # Pops happen externally (not through enqueue()), so the cache is stale
        # after processing; this one-liner restores accuracy before the update().
        self.rate_limiter.stream_backlog = len(self.rate_limiter.event_queue)

        applied = self.rate_limiter.update(
            elapsed=elapsed,
            stream_rate=self.stream_rate,
            latest_drift=latest_drift,
            is_retraining=is_retraining,
            sample_index=self.sample_index,
            audit=self.audit,
        )

        if self.rate_limiter.state in {"Throttling", "Protecting"}:
            prev_sample = self.rate_limiter._last_audit_sample
            if prev_sample == self.sample_index:
                self.action_reasons["Rate control"] += 1

        return applied

    # -----------------------------------------------------------------------
    # Retraining pipeline: performance gate -> shadow evaluation
    # -----------------------------------------------------------------------

    def _maybe_retrain(self, action, drift_score, rolling_avg):
        should_retrain_now, required, elapsed = self.cooldown.should_retrain(
            self.sample_index, drift_score
        )

        if self.shadow_evaluator.is_evaluating or self.retrain_job_active:
            return required, elapsed

        strong_signal = action in {"RETRAIN", "RETRAIN_URGENT"} or drift_score > 0.68
        retrain_signal = (rolling_avg is not None and rolling_avg > 45) or drift_score > 0.72

        if not (strong_signal and should_retrain_now and retrain_signal and len(self.buffer) >= 55):
            return required, elapsed

        buffer_list = list(self.buffer)
        production_model = self.model
        production_scaler = self.scaler
        sample_index = self.sample_index

        self.retrain_job_active = True
        self.cooldown.mark_retrain(sample_index)
        self.system_state = "Self-healing"
        self.last_action = "Predictive model refresh initiated - training on recent field data"
        self.action_reasons["Background retrain"] += 1
        mae_str = f"{rolling_avg:.1f}" if rolling_avg is not None else "N/A"
        self.audit.append_timeline("Predictive model refresh initiated", "Warning")
        self.audit.append_audit(
            "Model Refresh Initiated",
            f"Cumulative drift index {drift_score:.2f} exceeds adaptive threshold - retraining on recent field observations",
            f"Candidate model training in background | Rolling prediction error: {mae_str} cycles",
            "Warning",
            "MODEL",
        )

        self.retrain_thread = Thread(
            target=self._run_retrain_job,
            args=(buffer_list, production_model, production_scaler),
            daemon=True,
        )
        self.retrain_thread.start()
        return required, elapsed

    def _run_retrain_job(self, buffer_list, production_model, production_scaler):
        try:
            result = self._train_retrain_candidate(
                buffer_list, production_model, production_scaler
            )
            self._apply_retrain_result(result)
        finally:
            with self.lock:
                self.retrain_job_active = False

    def _train_retrain_candidate(self, buffer_list, production_model, production_scaler):
        try:
            buffer_df = pd.DataFrame(buffer_list)
        except Exception:
            return {
                "accepted": False,
                "event": "Retrain Skipped",
                "reason": "Unable to build training frame from buffer",
                "action": "Kept current production model",
                "status": "Warning",
            }

        missing = [c for c in self.expected_train_columns if c not in buffer_df.columns]
        if missing:
            return {
                "accepted": False,
                "event": "Retrain Skipped",
                "reason": f"Schema mismatch - missing columns: {missing[:3]}",
                "action": "Kept current production model",
                "status": "Warning",
            }

        buffer_df = buffer_df[
            [c for c in self.expected_train_columns if c in buffer_df.columns]
        ]

        try:
            new_model, new_scaler, new_mae = train_model_with_holdout(
                buffer_df, min_retrain_rows=30
            )
        except Exception as err:
            return {
                "accepted": False,
                "event": "Retrain Failed",
                "reason": f"Training error: {str(err)[:120]}",
                "action": "Kept current production model",
                "status": "Warning",
            }

        if new_model is None:
            return {
                "accepted": False,
                "event": "Retrain Skipped",
                "reason": "Candidate model was not produced",
                "action": "Kept current production model",
                "status": "Warning",
            }

        validation_df = pd.DataFrame(buffer_list[-20:])
        validation_df = validation_df[
            [c for c in self.expected_train_columns if c in validation_df.columns]
        ]
        should_accept, current_mae, candidate_mae, gate_reason = (
            self.performance_gate.should_accept_new_model(
                production_model,
                production_scaler,
                new_model,
                new_scaler,
                new_mae,
                validation_df,
            )
        )

        return {
            "accepted": should_accept,
            "event": "Shadow Started" if should_accept else "Retrain Rejected",
            "reason": gate_reason,
            "action": "Running candidate in shadow evaluation"
            if should_accept
            else "Production model retained - no shadow evaluation",
            "status": "Warning",
            "new_model": new_model,
            "new_scaler": new_scaler,
            "current_mae": current_mae,
            "candidate_mae": candidate_mae,
        }

    def _apply_retrain_result(self, result):
        cand = result.get("candidate_mae")
        curr = result.get("current_mae")
        cand_str = f"{cand:.2f}" if cand is not None else "N/A"
        curr_str = f"{curr:.2f}" if curr is not None else "N/A"

        with self.lock:
            if (
                result.get("accepted")
                and result.get("new_model") is not None
                and result.get("new_scaler") is not None
                and not self.shadow_evaluator.is_evaluating
            ):
                self.shadow_evaluator.start_shadow_evaluation(
                    result["new_model"], result["new_scaler"]
                )
                self.system_state = "Shadowing"
                self.last_action = f"Candidate model under parallel evaluation - running alongside production"
                self.action_reasons["Shadow evaluation"] += 1
                self.audit.append_timeline("Candidate model in parallel evaluation", "Warning")
                self.audit.append_audit(
                    "Parallel Model Evaluation",
                    f"Candidate model passed quality gate (error reduction verified) - evaluating over {self.shadow_evaluator.window_size} live sensor cycles",
                    f"Candidate prediction error: {cand_str} cycles | Production prediction error: {curr_str} cycles",
                    result["status"],
                    "MODEL",
                )
                return

            self.action_reasons["Quality gate"] += 1
            gate_event = "Model Refresh Unsuccessful" if result["event"] == "Retrain Failed" else "Model Refresh Superseded"
            gate_detail = (
                f"Candidate prediction error ({cand_str} cycles) did not improve over production ({curr_str} cycles)"
                if cand_str != "N/A" and curr_str != "N/A"
                else result["reason"]
            )
            self.audit.append_audit(
                gate_event,
                gate_detail,
                "Production model retained - no change to active inference pipeline",
                result["status"],
                "MODEL",
            )
            if result["event"] in {"Retrain Failed", "Retrain Rejected"}:
                self.audit.append_alert(
                    "Warning",
                    f"Model refresh attempt did not meet quality threshold - production model continues unchanged"
                )

    # -----------------------------------------------------------------------
    # Per-sample processing loop
    # -----------------------------------------------------------------------

    def _create_stream_event(self):
        raw = self._next_row()
        data = self._apply_stream_noise(raw)
        scenario_active = self._apply_active_scenario(data)
        return {"data": data, "scenario_active": scenario_active}

    def _enqueue_stream_event(self):
        prev_shed = self.rate_limiter.load_shedding_total
        self.rate_limiter.enqueue(self._create_stream_event(), self.sample_index, self.audit)
        if self.rate_limiter.load_shedding_total > prev_shed:
            self.action_reasons["Load Shedding (Stale Data)"] += 1
        self.stream_backlog = len(self.rate_limiter.event_queue)

    def _warm_up_stream(self):
        for _ in range(INITIAL_STREAM_WARMUP_SAMPLES):
            self._enqueue_stream_event()
        self.current_incoming_rate = self.stream_rate
        self.current_processed_rate = min(self.stream_rate, self.rate_limiter.applied_rate_limit)
        started = time.perf_counter()
        processed = 0
        warmup_events = [
            self.rate_limiter.event_queue.popleft()
            for _ in range(min(INITIAL_STREAM_WARMUP_SAMPLES, len(self.rate_limiter.event_queue)))
        ]
        try:
            predictions = self._predict_batch_with_confidence(warmup_events)
        except Exception:
            predictions = [None] * len(warmup_events)
        anomaly_results = self._detect_anomalies_batch(warmup_events)
        for event, prediction, anomaly_result in zip(warmup_events, predictions, anomaly_results):
            self.stream_backlog = len(self.rate_limiter.event_queue)
            self._process_one(event, prediction, anomaly_result)
            processed += 1
        elapsed = time.perf_counter() - started
        if processed > 0 and elapsed > 0:
            measured_capacity = processed / elapsed
            self.rate_limiter.worker_capacity_limit = float(np.clip(measured_capacity * 0.92, 1.0, 40.0))

    def _process_one(self, event, prediction=None, anomaly_result=None):
        data = event["data"]
        scenario_active = event["scenario_active"]

        self.buffer.append(data.copy())
        self.current_feature_window.append(
            {f: data[f] for f in FEATURE_COLUMNS}
        )

        actual = float(data["RUL"])

        # --- Isolation Forest anomaly detection ---
        if anomaly_result is not None:
            is_anomaly, anomaly_score = anomaly_result
        else:
            try:
                is_anomaly, anomaly_score = self.anomaly_detector.is_anomaly(data)
            except Exception:
                is_anomaly, anomaly_score = False, 0.0

        # --- RF prediction with confidence intervals ---
        try:
            if prediction is None:
                prediction = self._predict_with_confidence_fast(data)
            pred, pred_lower, pred_upper, pred_std = prediction
            self.last_prediction_std = pred_std
        except Exception:
            pred = float(actual)
            pred_lower = pred - 10.0
            pred_upper = pred + 10.0
            pred_std = 5.0

        confidence_category = self.confidence_predictor.get_confidence_category(pred_std)
        error = abs(actual - pred)

        # --- Rolling error monitoring ---
        rolling_avg = self.error_monitor.update(error)
        trend = self.error_monitor.is_increasing()

        # --- Shadow model A/B evaluation ---
        if self.shadow_evaluator.is_evaluating:
            should_promote, prod_mae, shadow_mae = False, None, None
            try:
                shadow_pred = self._predict_model_fast(
                    self.shadow_evaluator.shadow_model,
                    self.shadow_evaluator.shadow_scaler,
                    data,
                )
                self.shadow_evaluator.production_errors.append(error)
                self.shadow_evaluator.shadow_errors.append(abs(actual - shadow_pred))
                if len(self.shadow_evaluator.shadow_errors) >= self.shadow_evaluator.window_size:
                    prod_mae = float(np.mean(self.shadow_evaluator.production_errors))
                    shadow_mae = float(np.mean(self.shadow_evaluator.shadow_errors))
                    should_promote = shadow_mae < (
                        prod_mae * self.shadow_evaluator.improvement_threshold
                    )
            except Exception:
                should_promote, prod_mae, shadow_mae = False, None, None

            if should_promote:
                self.model = self.shadow_evaluator.shadow_model
                self.scaler = self.shadow_evaluator.shadow_scaler
                self.model_version += 1
                self.shadow_evaluator.stop_evaluation()
                self.system_state = "Monitoring"
                self.last_action = (
                    f"Model updated to v1.0.{self.model_version} - improved accuracy confirmed"
                )
                self.action_reasons["Shadow promoted"] += 1
                self.audit.append_model_history(self.model_version)
                self.audit.append_timeline(
                    f"Model v1.0.{self.model_version} deployed - accuracy improvement confirmed",
                    "Healthy",
                )
                self.audit.append_audit(
                    "Model Updated",
                    f"Candidate model confirmed superior over {self.shadow_evaluator.window_size}-cycle evaluation window - prediction error reduced from {prod_mae:.1f} to {shadow_mae:.1f} cycles",
                    f"Active model promoted to v1.0.{self.model_version} | Monitoring resumed",
                    "Healthy",
                    "MODEL",
                )
                try:
                    if len(self.buffer) >= 30:
                        buffer_df = pd.DataFrame(list(self.buffer))
                        self.anomaly_detector.fit(buffer_df)
                except Exception:
                    pass
            elif prod_mae is not None:
                self.shadow_evaluator.stop_evaluation()
                self.system_state = "Monitoring"
                self.last_action = "Candidate model evaluation complete - production model retained"
                self.action_reasons["Shadow rejected"] += 1
                self.audit.append_timeline(
                    "Model evaluation complete - production model retained", "Warning"
                )
                self.audit.append_audit(
                    "Model Refresh Superseded",
                    f"Candidate did not demonstrate sufficient accuracy improvement over {self.shadow_evaluator.window_size}-cycle window (error: {shadow_mae:.1f} vs {prod_mae:.1f} cycles)",
                    "Production model retained - monitoring continues at standard thresholds",
                    "Warning",
                    "MODEL",
                )

        # --- ADWIN concept drift detection (error stream) ---
        adwin_drift = self.adwin_detector.update(
            rolling_avg if rolling_avg is not None else error
        )

        # --- KS-test feature drift detection ---
        drift_result = self.data_drift_detector.update_with_details(data)
        data_drift = bool(drift_result["drift_detected"])
        ks_drift_score = float(drift_result["drift_score"])
        combined_drift = adwin_drift or data_drift

        # --- Composite drift score ---
        drift_score = ks_drift_score
        if rolling_avg is not None:
            drift_score = max(drift_score, min(rolling_avg / 70.0, 1.0) * 0.65)
        if adwin_drift:
            drift_score = max(drift_score, 0.72)
        drift_score = float(drift_score)

        # --- Decision engine ---
        action = self.engine.decide(combined_drift, rolling_avg, trend, drift_score)
        required_cooldown, elapsed_cooldown = self._maybe_retrain(
            action, drift_score, rolling_avg
        )

        # --- Confidence and status ---
        confidence = confidence_value(pred_std, error)
        self.confidences.append(confidence)
        status = status_from_metrics(drift_score, rolling_avg, action)

        # Update system state
        if not self.shadow_evaluator.is_evaluating:
            if action.startswith("RETRAIN"):
                self.system_state = "Self-healing"
            elif self.system_state != "Shadowing":
                self.system_state = "Monitoring"

        # --- Anomaly condition audit (throttled) ---
        if scenario_active and self.active_scenario_cycle % 12 == 0:
            sc = SCENARIO_REGISTRY.get(self.active_scenario_id)
            if sc:
                self.action_reasons["Anomaly condition"] += 1
                severity_label = "Critical" if drift_score > 0.8 else "Warning"
                self.audit.append_alert(
                    severity_label,
                    f"Abnormal sensor behavior persisting - drift index {drift_score:.2f} | {self.active_scenario_remaining} observation cycles remaining",
                )
                self.audit.append_audit(
                    "Persistent Sensor Anomaly",
                    f"Anomalous pattern sustained across {self.active_scenario_cycle} consecutive cycles - statistical drift confirmed",
                    "Drift monitors at heightened sensitivity - automatic model assessment triggered",
                    status,
                    "DRIFT",
                )

        # --- Decision audit (throttled) ---
        if action in {"RETRAIN", "RETRAIN_URGENT", "ALERT", "MONITOR"} and self.sample_index % 10 == 0:
            event_type = "DRIFT" if action.startswith("RETRAIN") else "CONFIDENCE"
            drift_channels = sum([int(adwin_drift), int(data_drift)])
            if action == "RETRAIN_URGENT":
                timeline_event = "Degradation pattern escalating - urgent model refresh queued"
                audit_event = "Degradation Pattern Escalation"
                audit_reason = (
                    f"Drift index {drift_score:.2f} - statistical shift confirmed across {drift_channels} detection channel(s)"
                    f" | Prediction error trending {'upward' if trend else 'stable'}"
                )
                audit_action = "Autonomous health policy has escalated - model refresh queued at next eligible window"
            elif action == "RETRAIN":
                timeline_event = "Sensor drift exceeds threshold - model refresh scheduled"
                audit_event = "Predictive Health Degradation"
                audit_reason = (
                    f"Drift index {drift_score:.2f} - {drift_channels} detection channel(s) signaling distributional shift"
                    f" | Current prediction error: {error:.1f} cycles"
                )
                audit_action = "Autonomous health policy scheduling model refresh when cooldown elapses"
            elif action == "ALERT":
                timeline_event = "Prediction confidence degraded - operator review advised"
                audit_event = "Confidence Threshold Breach"
                audit_reason = (
                    f"Prediction confidence below operational threshold - error {error:.1f} cycles"
                    f" | Drift index: {drift_score:.2f}"
                )
                audit_action = "Operator review advised - system continues autonomous monitoring"
            else:  # MONITOR
                timeline_event = "Elevated drift observed - monitoring at increased frequency"
                audit_event = "Elevated Sensor Drift"
                audit_reason = (
                    f"Drift index {drift_score:.2f} approaching alert threshold - {drift_channels} channel(s) flagged"
                )
                audit_action = "No intervention required - monitoring cadence increased"
            self.audit.append_timeline(timeline_event, status)
            self.audit.append_audit(
                audit_event,
                audit_reason,
                audit_action,
                status,
                event_type,
            )
            self.action_reasons[
                "Drift" if event_type == "DRIFT" else "Confidence drop"
            ] += 1

        clock, _ = self._now_parts()
        latency = int(
            np.clip(
                12
                + self.current_processed_rate * 1.5
                + drift_score * 80
                + self.stream_backlog / 2.8,
                10,
                420,
            )
        )

        point = {
            "time": clock,
            "sampleIndex": self.sample_index,
            "dataRate": round(self.current_incoming_rate, 1),
            "incomingRate": round(self.current_incoming_rate, 1),
            "processedRate": round(self.current_processed_rate, 1),
            "appliedRateLimit": round(self.rate_limiter.applied_rate_limit, 1),
            "configuredRateLimit": round(self.rate_limiter.rate_limit, 1),
            "throttledRate": round(
                max(0.0, self.current_incoming_rate - self.current_processed_rate), 1
            ),
            "rateControlState": self.rate_limiter.state,
            "latency": latency,
            "driftScore": round(drift_score, 3),
            "confidence": confidence,
            "confidenceCategory": confidence_category,
            "error": round(min(error / 125.0, 1.0), 3),
            "mae": round(error, 2),
            "rollingMae": None if rolling_avg is None else round(float(rolling_avg), 2),
            "accuracy": round(max(0.0, 1.0 - min(error / 125.0, 1.0)), 3),
            "streamBacklog": int(self.stream_backlog),
            "prediction": round(pred, 2),
            "actual": round(actual, 2),
            "predLower": round(pred_lower, 2),
            "predUpper": round(pred_upper, 2),
            "action": action,
            "status": status,
            "adwinDrift": bool(adwin_drift),
            "dataDrift": bool(data_drift),
            "anomalyDetected": bool(is_anomaly),
            "anomalyScore": round(float(anomaly_score), 3),
            "scenarioActive": scenario_active,
            "cooldownElapsed": min(elapsed_cooldown, required_cooldown),
            "cooldownRequired": required_cooldown,
        }
        self.latest_point = point
        self.sample_index += 1

    # -----------------------------------------------------------------------
    # Background stream worker and dashboard snapshots
    # -----------------------------------------------------------------------

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.stop_event.clear()
        self.worker_thread = Thread(target=self._run_stream_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)

    def _run_stream_loop(self):
        self.last_advance = time.monotonic()
        self.last_snapshot_at = self.last_advance

        while not self.stop_event.is_set():
            tick_started = time.monotonic()
            with self.lock:
                self._stream_tick(tick_started)

            elapsed = time.monotonic() - tick_started
            sleep_for = max(0.0, STREAM_TICK_SECONDS - elapsed)
            self.stop_event.wait(sleep_for)

    def _stream_tick(self, now):
        elapsed = max(0.0, now - self.last_advance)
        if elapsed <= 0:
            return

        incoming_float = elapsed * self.stream_rate + self.arrival_carry
        incoming_events = int(incoming_float)
        self.arrival_carry = incoming_float - incoming_events
        for _ in range(incoming_events):
            self._enqueue_stream_event()

        applied_limit = self._update_rate_controller(elapsed)
        self.processing_carry = min(
            self.processing_carry + elapsed * applied_limit,
            MAX_PROCESSING_BURST_EVENTS,
        )
        process_due = min(
            int(self.processing_carry),
            len(self.rate_limiter.event_queue),
            MAX_EVENTS_PER_WORKER_TICK,
        )
        self.processing_carry -= process_due

        batch = [self.rate_limiter.event_queue.popleft() for _ in range(process_due)]
        try:
            predictions = self._predict_batch_with_confidence(batch)
        except Exception:
            predictions = [None] * len(batch)
        anomaly_results = self._detect_anomalies_batch(batch)

        for event, prediction, anomaly_result in zip(batch, predictions, anomaly_results):
            self.stream_backlog = len(self.rate_limiter.event_queue)
            self._process_one(event, prediction, anomaly_result)

        self.incoming_events_since_snapshot += incoming_events
        self.processed_events_since_snapshot += process_due
        self.stream_backlog = len(self.rate_limiter.event_queue)
        self.last_advance = now

        if now - self.last_snapshot_at >= DASHBOARD_SNAPSHOT_SECONDS:
            self._publish_dashboard_snapshot(now)

    def _publish_dashboard_snapshot(self, now=None, update_rates=True):
        now = time.monotonic() if now is None else now
        if update_rates:
            rate_window = max(now - self.last_snapshot_at, DASHBOARD_SNAPSHOT_SECONDS)
            self.current_incoming_rate = self.incoming_events_since_snapshot / rate_window
            self.current_processed_rate = self.processed_events_since_snapshot / rate_window

            self.incoming_events_since_snapshot = 0
            self.processed_events_since_snapshot = 0
            self.last_snapshot_at = now

        if update_rates and self.latest_point is not None:
            point = {
                **self.latest_point,
                "time": self._now_parts()[0],
                "dataRate": round(self.current_incoming_rate, 1),
                "incomingRate": round(self.current_incoming_rate, 1),
                "processedRate": round(self.current_processed_rate, 1),
                "appliedRateLimit": round(self.rate_limiter.applied_rate_limit, 1),
                "configuredRateLimit": round(self.rate_limiter.rate_limit, 1),
                "throttledRate": round(
                    max(0.0, self.current_incoming_rate - self.current_processed_rate), 1
                ),
                "rateControlState": self.rate_limiter.state,
                "streamBacklog": int(self.stream_backlog),
            }
            self.latest_point = point
            self.series.append(point)

        snapshot = self._build_dashboard_response()
        with self.snapshot_lock:
            self.dashboard_snapshot = snapshot
        if update_rates:
            self.rate_limiter.reset_snapshot_counters()

    def _build_dashboard_response(self):
        latest = self.latest_point or (self.series[-1] if self.series else {})
        feature_scores = self._feature_scores()
        series = list(self.series)
        action_reasons = (
            [
                {"reason": r, "value": v}
                for r, v in self.action_reasons.items()
                if v > 0
            ]
            or [{"reason": "Monitoring", "value": 1}]
        )

        # Active scenario metadata for dashboard
        sc_meta = (
            SCENARIO_REGISTRY[self.active_scenario_id].META
            if self.active_scenario_id
            else None
        )

        # Compute overall system status considering both ML health and infrastructure state
        ml_status = latest.get("status", "Healthy")
        if self.stream_backlog >= STREAM_QUEUE_MAXLEN or self.rate_limiter.load_shedding_events_since_snapshot > 0:
            system_status = "Critical"
        elif self.rate_limiter.state == "Throttling":
            system_status = "Critical" if ml_status == "Critical" else "Warning"
        else:
            system_status = ml_status

        return {
            "overview": {
                "systemStatus": system_status,
                "activeModelVersion": f"v1.0.{self.model_version}",
                "series": series,
                "confidenceHistogram": self._histogram(),
                "latestSample": {
                    "prediction": latest.get("prediction", 0),
                    "actual": latest.get("actual", 0),
                    "interval": [
                        latest.get("predLower", 0),
                        latest.get("predUpper", 0),
                    ],
                    "action": latest.get("action", "STABLE"),
                    "sampleIndex": latest.get("sampleIndex", 0),
                    "confidenceCategory": latest.get(
                        "confidenceCategory", "medium_confidence"
                    ),
                    "adwinDrift": latest.get("adwinDrift", False),
                    "dataDrift": latest.get("dataDrift", False),
                    "anomalyDetected": latest.get("anomalyDetected", False),
                    "anomalyScore": latest.get("anomalyScore", 0.0),
                },
                "meta": {
                    "live": not self.stop_event.is_set(),
                    "snapshotIntervalMs": int(DASHBOARD_SNAPSHOT_SECONDS * 1000),
                    "workerTickMs": int(STREAM_TICK_SECONDS * 1000),
                    "lastUpdatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                "scenario": {
                    "active": self.active_scenario_remaining > 0,
                    "remaining": self.active_scenario_remaining,
                    "id": self.active_scenario_id,
                    "name": sc_meta["name"] if sc_meta else None,
                    "severity": sc_meta["severity"] if sc_meta else None,
                },
            },
            "drift": {
                "featureDistributionShift": self._distribution_shift(feature_scores),
                "featureScores": feature_scores,
                "conceptSeries": series,
                "alerts": self.audit.get_alerts(),
            },
            "rateControl": {
                "rateLimitEnabled": self.rate_limiter.rate_limit_enabled,
                "rateLimit": round(self.rate_limiter.rate_limit, 1),
                "simulatedRate": round(self.stream_rate, 1),
                "incomingRate": round(self.current_incoming_rate, 1),
                "currentRate": round(self.current_processed_rate, 1),
                "processedRate": round(self.current_processed_rate, 1),
                "appliedRateLimit": round(self.rate_limiter.applied_rate_limit, 1),
                "throttledRate": round(
                    max(0.0, self.current_incoming_rate - self.current_processed_rate),
                    1,
                ),
                "streamBacklog": int(self.stream_backlog),
                "queueCapacity": STREAM_QUEUE_MAXLEN,
                "loadSheddingActive": (
                    self.rate_limiter.load_shedding_events_since_snapshot > 0
                    or (
                        self.rate_limiter.last_load_shedding_at is not None
                        and time.monotonic() - self.rate_limiter.last_load_shedding_at <= 4.0
                    )
                ),
                "loadSheddingTotal": int(self.rate_limiter.load_shedding_total),
                "loadSheddingRecent": int(self.rate_limiter.load_shedding_events_since_snapshot),
                "workerCapacity": round(self.rate_limiter.worker_capacity_limit, 1),
                "controllerState": self.rate_limiter.state,
                "controllerReason": self.rate_limiter.reason,
                "overloadRisk": int(
                    np.clip(
                        max(self.stream_backlog / 90.0, latest.get("latency", 45) / 260.0)
                        * 100,
                        0,
                        100,
                    )
                ),
                "cpuUsage": int(
                    np.clip(
                        12 + self.current_processed_rate * 1.8 + latest.get("driftScore", 0.0) * 24,
                        5,
                        96,
                    )
                ),
                "memoryUsage": int(
                    np.clip(
                        22 + self.stream_backlog / 5.0 + latest.get("driftScore", 0.0) * 18,
                        15,
                        92,
                    )
                ),
                "actualVsLimitSeries": [
                    {
                        "time": p["time"],
                        "sampleIndex": p.get("sampleIndex", 0),
                        "incomingRate": p.get("incomingRate", p["dataRate"]),
                        "actualRate": p.get("processedRate", p["dataRate"]),
                        "appliedRateLimit": p.get("appliedRateLimit", round(self.rate_limiter.applied_rate_limit, 1)),
                        "rateLimit": p.get("configuredRateLimit", round(self.rate_limiter.rate_limit, 1)),
                    }
                    for p in series
                ],
                "streamBacklogSeries": [
                    {
                        "time": p["time"],
                        "sampleIndex": p.get("sampleIndex", 0),
                        "streamBacklog": p["streamBacklog"],
                    }
                    for p in series
                ],
            },
            "selfHealing": {
                "timeline": self.audit.get_timeline(),
                "modelHistory": self.audit.get_model_history(),
                "actionReasons": action_reasons,
                "lastActionTaken": self.last_action,
                "currentSystemState": self.system_state,
                "shadowEvaluation": self.shadow_evaluator.get_status(),
            },
            "auditLogs": self.audit.get_audit_logs(),
        }

    # -----------------------------------------------------------------------
    # Public API methods
    # -----------------------------------------------------------------------

    def inject_anomaly(self, request):
        with self.lock:
            scenario_cls = SCENARIO_REGISTRY.get(request.scenario)
            if scenario_cls is None:
                return {"ok": False, "error": f"Unknown scenario: {request.scenario}"}
            meta = scenario_cls.META
            self.active_scenario_id = request.scenario
            self.active_scenario_remaining = meta["duration"]
            self.active_scenario_cycle = 0
            self.last_action = (
                f"Fault condition activated - drift monitors and health policy on heightened alert"
            )
            self.system_state = "Self-healing"
            alert_severity = "Critical" if meta["severity"] in {"Critical", "High"} else "Warning"
            self.audit.append_timeline("Fault condition introduced - monitors escalated", "Critical")
            self.audit.append_audit(
                "Fault Scenario Activated",
                meta["description"],
                f"Duration: {meta['duration']} observation cycles | Severity class: {meta['severity']}",
                "Critical",
                "DRIFT",
            )
            self.audit.append_alert(
                alert_severity,
                f"Anomalous sensor pattern introduced - drift detection and health policy on heightened alert",
            )
            return {"ok": True, "scenario": meta}

    def update_controls(self, request):
        rate_control_patch = {}
        with self.lock:
            if request.simulatedRate is not None:
                self.stream_rate = float(request.simulatedRate)
            self.rate_limiter.apply_controls(
                rate_limit=request.rateLimit,
                rate_limit_enabled=request.rateLimitEnabled,
            )
            rate_control_patch = {
                "rateLimitEnabled": self.rate_limiter.rate_limit_enabled,
                "rateLimit": round(self.rate_limiter.rate_limit, 1),
                "simulatedRate": round(self.stream_rate, 1),
                "appliedRateLimit": round(self.rate_limiter.applied_rate_limit, 1),
                "controllerState": self.rate_limiter.state,
                "controllerReason": self.rate_limiter.reason,
            }
            current_key = (round(self.stream_rate, 1), round(self.rate_limiter.rate_limit, 1), self.rate_limiter.rate_limit_enabled)
            if current_key != self._last_controls_audit:
                self._last_controls_audit = current_key
                ceiling_str = (
                    f"{self.rate_limiter.rate_limit:.1f} eps"
                    if self.rate_limiter.rate_limit_enabled
                    else f"{self.rate_limiter.worker_capacity_limit:.1f} eps (hardware limit)"
                )
                status_str = "Enabled" if self.rate_limiter.rate_limit_enabled else "Disabled"
                self.audit.append_audit(
                    "Ingestion Policy Updated",
                    "Operator modified sensor stream processing controls",
                    f"Rate limit: {status_str} | Configured ingestion rate: {self.stream_rate:.1f} eps | Rate ceiling: {ceiling_str}",
                    "Healthy",
                    "RATE_LIMIT",
                )
        with self.snapshot_lock:
            if self.dashboard_snapshot is not None:
                self.dashboard_snapshot["rateControl"].update(rate_control_patch)
        return {"ok": True}

    def dashboard(self):
        with self.snapshot_lock:
            return self.dashboard_snapshot


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

runtime = StreamingMLRuntime()
app = FastAPI(title="Autonomous Self-Healing ML API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "modelVersion": f"v1.0.{runtime.model_version}"}


@app.on_event("shutdown")
def shutdown_runtime():
    runtime.stop()


@app.get("/api/dashboard")
def dashboard():
    return runtime.dashboard()


@app.get("/api/scenarios")
def get_scenarios():
    return {"scenarios": get_scenario_list()}


@app.post("/api/anomalies")
def inject_anomaly(request: AnomalyRequest):
    return runtime.inject_anomaly(request)


@app.post("/api/controls")
def update_controls(request: ControlRequest):
    return runtime.update_controls(request)


@app.post("/api/reset")
def reset_runtime():
    global runtime
    old_runtime = runtime
    old_runtime.stop()
    runtime = StreamingMLRuntime()
    return {"ok": True}
