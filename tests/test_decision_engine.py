from decision.engine import DecisionEngine


def test_decision_engine_retrain_on_drift():
    engine = DecisionEngine(error_threshold=40)
    assert engine.decide(drift=True, rolling_avg=10, trend=False) == "RETRAIN"


def test_decision_engine_retrain_urgent_on_severe_drift():
    engine = DecisionEngine(error_threshold=40)
    assert engine.decide(drift=True, rolling_avg=10, trend=False, drift_score=0.9) == "RETRAIN_URGENT"


def test_decision_engine_monitor_on_high_increasing_error():
    engine = DecisionEngine(error_threshold=40)
    assert engine.decide(drift=False, rolling_avg=45, trend=True) == "MONITOR"


def test_decision_engine_alert_on_critically_high_error():
    engine = DecisionEngine(error_threshold=40)
    assert engine.decide(drift=False, rolling_avg=70, trend=False) == "ALERT"


def test_decision_engine_watch_when_error_approaching_threshold():
    engine = DecisionEngine(error_threshold=40)
    # 36 > 40 * 0.8 = 32, no trend, no drift
    assert engine.decide(drift=False, rolling_avg=36, trend=False) == "WATCH"


def test_decision_engine_stable_otherwise():
    engine = DecisionEngine(error_threshold=40)
    assert engine.decide(drift=False, rolling_avg=None, trend=False) == "STABLE"
