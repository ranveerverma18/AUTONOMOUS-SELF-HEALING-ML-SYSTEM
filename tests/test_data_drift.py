from drift.data_drift import DataDriftDetector


def _make_stable_point(value=1.0):
    return {f"sensor_{i}": value for i in range(1, 10)}


def test_warmup_phase_returns_no_drift():
    detector = DataDriftDetector(window_size=10)
    for _ in range(9):
        result = detector.update_with_details(_make_stable_point())
        assert result["drift_detected"] is False
        assert result["phase"] in ("reference_warmup", "current_warmup")


def test_stable_data_does_not_trigger_drift():
    detector = DataDriftDetector(window_size=10, drift_feature_ratio_threshold=0.3)
    # Fill reference window
    for _ in range(10):
        detector.update_with_details(_make_stable_point(1.0))
    # Fill current window with same distribution
    for _ in range(10):
        result = detector.update_with_details(_make_stable_point(1.0))
    assert result["drift_detected"] is False


def test_large_shift_triggers_drift():
    detector = DataDriftDetector(
        window_size=10,
        p_threshold=0.05,
        drift_feature_ratio_threshold=0.3,
        min_effect_size=0.05,
    )
    # Reference: all zeros
    for _ in range(10):
        detector.update_with_details(_make_stable_point(0.0))
    # Current: all very large values to force KS drift
    result = None
    for _ in range(10):
        result = detector.update_with_details(_make_stable_point(1000.0))
    assert result["drift_detected"] is True


def test_update_shortcut_returns_bool():
    detector = DataDriftDetector(window_size=5)
    result = detector.update(_make_stable_point())
    assert isinstance(result, bool)


def test_invalid_point_returns_no_drift():
    detector = DataDriftDetector(window_size=5)
    result = detector.update_with_details({"non_numeric": "oops"})
    assert result["drift_detected"] is False
    assert result["phase"] == "invalid_point"
