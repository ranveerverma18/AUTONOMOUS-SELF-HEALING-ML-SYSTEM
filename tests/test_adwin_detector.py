from drift.adwin_detector import DriftDetector


def test_no_drift_on_stable_stream():
    detector = DriftDetector()
    for _ in range(200):
        triggered = detector.update(1.0)
    assert triggered is False


def test_drift_detected_after_step_change():
    detector = DriftDetector()
    # Feed stable values first
    for _ in range(100):
        detector.update(0.0)
    # Then inject a large sudden shift
    drift_seen = False
    for _ in range(200):
        if detector.update(100.0):
            drift_seen = True
            break
    assert drift_seen is True


def test_update_returns_bool():
    detector = DriftDetector()
    result = detector.update(5.0)
    assert isinstance(result, bool)
