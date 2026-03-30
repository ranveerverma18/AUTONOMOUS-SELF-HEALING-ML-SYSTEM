from drift.adwin_detector import DriftDetector


def test_drift_detector_update_returns_bool():
    detector = DriftDetector()
    out = detector.update(0.1)
    assert isinstance(out, bool)
