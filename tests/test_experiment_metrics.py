from experiments.metrics import summarize_events


def _row(
    i,
    error,
    feature=False,
    concept=False,
    anomaly=False,
    promotion=None,
    retraining_triggered=None,
    candidate_generated=False,
    gate_passed=None,
    gate_rejected=False,
    shadow_passed=False,
    shadow_rejected=False,
    model_promoted=False,
    validation_skipped=False,
    validation_skip_reason=None,
):
    return {
        "sample_index": i,
        "absolute_error": error,
        "squared_error": error**2,
        "rolling_mae": error,
        "feature_drift_detected": feature,
        "concept_drift_detected": concept,
        "anomaly_detected": anomaly,
        "retraining_triggered": (i == 3) if retraining_triggered is None else retraining_triggered,
        "retraining_completed": i == 3,
        "validation_skipped": validation_skipped,
        "validation_skip_reason": validation_skip_reason,
        "candidate_generated": candidate_generated,
        "candidate_id": "candidate-1" if candidate_generated else None,
        "gate_passed": gate_passed,
        "gate_rejected": gate_rejected,
        "shadow_passed": shadow_passed,
        "shadow_rejected": shadow_rejected,
        "model_promoted": model_promoted,
        "promotion_decision": promotion,
        "degraded_promotion": False,
        "training_time": 0.5 if i == 3 else 0.0,
        "shadow_evaluation_time": 2.0 if promotion else 0.0,
        "inference_latency": 0.01,
    }


def test_summary_metrics_are_computed():
    rows = [
        _row(0, 1.0),
        _row(1, 2.0, feature=True),
        _row(2, 3.0, anomaly=True),
        _row(
            3,
            4.0,
            concept=True,
            promotion="promoted",
            candidate_generated=True,
            gate_passed=True,
            shadow_passed=True,
            model_promoted=True,
        ),
    ]

    summary = summarize_events(rows, scenario_start_index=1, recovery_error_threshold=2.5)

    assert summary.mae == 2.5
    assert round(summary.rmse, 3) == 2.739
    assert summary.detection_delay == 0
    assert summary.drift_detections == 2
    assert summary.anomaly_detections == 1
    assert summary.false_positive_triggers == 0
    assert summary.time_to_first_error_recovery is None
    assert summary.candidates_generated == 1
    assert summary.shadow_promotions == 1
    assert summary.model_promoted_events == 1
    assert summary.gate_accepts == 1
    assert summary.gate_rejects == 0
    assert summary.validation_skipped_events == 0


def test_validation_skipped_events_counted_separately():
    rows = [
        _row(0, 5.0),
        _row(
            1,
            5.0,
            retraining_triggered=True,
            validation_skipped=True,
            validation_skip_reason="validation_too_small:10_rows",
        ),
        _row(
            2,
            5.0,
            retraining_triggered=True,
            candidate_generated=True,
            gate_passed=True,
            shadow_passed=True,
            model_promoted=True,
            promotion="promoted",
        ),
    ]
    summary = summarize_events(rows, scenario_start_index=0)

    assert summary.retraining_events == 2
    assert summary.validation_skipped_events == 1
    assert summary.candidates_generated == 1
    assert summary.model_promoted_events == 1


def test_gate_rejects_not_counted_as_promotions():
    rows = [
        _row(0, 10.0),
        _row(
            1,
            10.0,
            retraining_triggered=True,
            candidate_generated=True,
            gate_passed=False,
            gate_rejected=True,
            promotion="gate_rejected:insufficient_improvement_0.5%",
        ),
    ]
    summary = summarize_events(rows, scenario_start_index=0)

    assert summary.candidates_generated == 1
    assert summary.gate_accepts == 0
    assert summary.gate_rejects == 1
    assert summary.shadow_promotions == 0
    assert summary.model_promoted_events == 0


def test_time_to_first_error_recovery_is_measured():
    rows = [
        _row(0, 10.0, retraining_triggered=True),
        _row(1, 60.0),
        _row(2, 60.0),
        _row(3, 20.0),
    ]
    summary = summarize_events(rows, scenario_start_index=0, recovery_error_threshold=45.0)
    assert summary.time_to_first_error_recovery == 2


def test_time_to_sustained_recovery_requires_consecutive_samples():
    rows = [
        _row(0, 10.0, retraining_triggered=True),
        _row(1, 60.0),
        _row(2, 20.0),
        _row(3, 60.0),
        _row(4, 20.0),
        _row(5, 20.0),
        _row(6, 20.0),
    ]
    summary = summarize_events(
        rows,
        scenario_start_index=0,
        recovery_error_threshold=45.0,
        sustained_recovery_window=3,
    )
    assert summary.time_to_sustained_recovery is not None
    assert summary.time_to_sustained_recovery == 5


def test_time_to_sustained_recovery_none_if_not_sustained():
    rows = [
        _row(0, 10.0, retraining_triggered=True),
        _row(1, 60.0),
        _row(2, 20.0),
        _row(3, 60.0),
    ]
    summary = summarize_events(
        rows,
        scenario_start_index=0,
        recovery_error_threshold=45.0,
        sustained_recovery_window=3,
    )
    assert summary.time_to_sustained_recovery is None

