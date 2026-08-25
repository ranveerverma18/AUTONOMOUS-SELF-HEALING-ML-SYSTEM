"""Experiment event and run-level metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunSummary:
    mae: float | None
    rmse: float | None
    detection_delay: int | None
    drift_detections: int
    anomaly_detections: int
    false_positive_triggers: int
    # --- Adaptation lifecycle ---
    retraining_events: int
    """Number of times retraining was triggered (includes validation-skipped cases)."""
    validation_skipped_events: int
    """Triggers where validation-quality requirements were not met; no candidate generated."""
    candidates_generated: int
    """Number of candidates that passed the validation-quality gate and were trained."""
    gate_accepts: int
    """Candidates where the performance gate found sufficient improvement."""
    gate_rejects: int
    """Candidates where the performance gate rejected due to insufficient improvement."""
    shadow_promotions: int
    """Candidates that passed shadow evaluation and were promoted to production."""
    shadow_rejections: int
    """Candidates that entered shadow evaluation but failed and were rejected."""
    model_promoted_events: int
    """Total model promotions (should equal shadow_promotions for proposed strategy)."""
    degraded_promotions: int
    degraded_promotion_rate: float | None
    # --- Recovery ---
    time_to_first_error_recovery: int | None
    """Samples from the first rolling_mae exceedance (after the first adaptation
    trigger) to the first subsequent drop below the recovery threshold.
    This is an approximation; it reflects one recovery crossing, not sustained recovery."""
    time_to_sustained_recovery: int | None
    """Samples from the first exceedance to the first point where rolling_mae
    remains below the recovery threshold for the configured consecutive window.
    None if sustained recovery was not observed within the run."""
    # --- Efficiency ---
    total_retraining_time: float
    total_shadow_evaluation_time: float
    total_adaptation_time: float
    mean_inference_latency: float | None


def _time_to_sustained_recovery(
    events: list[dict],
    first_adaptation: int | None,
    recovery_threshold: float,
    sustained_window: int,
) -> int | None:
    """Return samples from first exceedance to first sustained-recovery point.

    Sustained recovery is defined as rolling_mae remaining below
    recovery_threshold for at least sustained_window consecutive observations
    after first exceeding that threshold.
    """
    if first_adaptation is None or sustained_window < 1:
        return None

    exceeded_at = next(
        (
            row["sample_index"]
            for row in events
            if row["sample_index"] >= first_adaptation
            and row.get("rolling_mae") not in {None, ""}
            and float(row["rolling_mae"]) >= recovery_threshold
        ),
        None,
    )
    if exceeded_at is None:
        return None

    post_exceedance = [
        row
        for row in events
        if row["sample_index"] > exceeded_at
        and row.get("rolling_mae") not in {None, ""}
    ]

    consecutive = 0
    for row in post_exceedance:
        if float(row["rolling_mae"]) < recovery_threshold:
            consecutive += 1
            if consecutive >= sustained_window:
                return int(row["sample_index"] - exceeded_at)
        else:
            consecutive = 0
    return None


def summarize_events(
    events: list[dict],
    scenario_start_index: int,
    recovery_error_threshold: float | None = None,
    sustained_recovery_window: int = 5,
) -> RunSummary:
    absolute_errors = [row["absolute_error"] for row in events if row["absolute_error"] is not None]
    squared_errors = [row["squared_error"] for row in events if row["squared_error"] is not None]
    first_detection = next(
        (
            row["sample_index"]
            for row in events
            if row["feature_drift_detected"] or row["concept_drift_detected"]
        ),
        None,
    )
    latencies = [row["inference_latency"] for row in events if row.get("inference_latency") is not None]
    first_adaptation = next(
        (row["sample_index"] for row in events if row["retraining_triggered"]),
        None,
    )

    promotions = [row for row in events if row.get("model_promoted")]
    degraded_promotions = sum(1 for row in promotions if row.get("degraded_promotion") is True)

    time_to_first: int | None = None
    time_to_sustained: int | None = None
    if recovery_error_threshold is not None and first_adaptation is not None:
        exceeded_at = next(
            (
                row["sample_index"]
                for row in events
                if row["sample_index"] >= first_adaptation
                and row.get("rolling_mae") not in {None, ""}
                and float(row["rolling_mae"]) >= recovery_error_threshold
            ),
            None,
        )
        if exceeded_at is not None:
            recovered_at = next(
                (
                    row["sample_index"]
                    for row in events
                    if row["sample_index"] > exceeded_at
                    and row.get("rolling_mae") not in {None, ""}
                    and float(row["rolling_mae"]) < recovery_error_threshold
                ),
                None,
            )
            if recovered_at is not None:
                time_to_first = int(recovered_at - exceeded_at)

        time_to_sustained = _time_to_sustained_recovery(
            events, first_adaptation, recovery_error_threshold, sustained_recovery_window
        )

    return RunSummary(
        mae=float(np.mean(absolute_errors)) if absolute_errors else None,
        rmse=float(np.mean(squared_errors) ** 0.5) if squared_errors else None,
        detection_delay=(
            int(first_detection - scenario_start_index)
            if first_detection is not None and first_detection >= scenario_start_index
            else None
        ),
        drift_detections=sum(
            1 for row in events if row["feature_drift_detected"] or row["concept_drift_detected"]
        ),
        anomaly_detections=sum(1 for row in events if row["anomaly_detected"]),
        false_positive_triggers=sum(
            1
            for row in events
            if row["retraining_triggered"] and row["sample_index"] < scenario_start_index
        ),
        retraining_events=sum(1 for row in events if row["retraining_triggered"]),
        validation_skipped_events=sum(1 for row in events if row.get("validation_skipped")),
        candidates_generated=sum(1 for row in events if row.get("candidate_generated")),
        gate_accepts=sum(1 for row in events if row.get("gate_passed") is True),
        gate_rejects=sum(1 for row in events if row.get("gate_rejected") is True),
        shadow_promotions=sum(1 for row in events if row.get("shadow_passed") is True),
        shadow_rejections=sum(1 for row in events if row.get("shadow_rejected") is True),
        model_promoted_events=sum(1 for row in events if row.get("model_promoted") is True),
        degraded_promotions=degraded_promotions,
        degraded_promotion_rate=(
            degraded_promotions / len(promotions) if promotions else None
        ),
        time_to_first_error_recovery=time_to_first,
        time_to_sustained_recovery=time_to_sustained,
        total_retraining_time=float(sum(row.get("training_time") or 0.0 for row in events)),
        total_shadow_evaluation_time=float(
            sum(row.get("shadow_evaluation_time") or 0.0 for row in events)
        ),
        total_adaptation_time=float(
            sum(
                (row.get("training_time") or 0.0) + (row.get("shadow_evaluation_time") or 0.0)
                for row in events
            )
        ),
        mean_inference_latency=float(np.mean(latencies)) if latencies else None,
    )
