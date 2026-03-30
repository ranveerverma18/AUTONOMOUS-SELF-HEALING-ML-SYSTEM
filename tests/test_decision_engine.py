from decision.engine import DecisionEngine


def test_decision_engine_retrain_on_drift():
    engine = DecisionEngine(error_threshold=40)
    assert engine.decide(drift=True, rolling_avg=10, trend=False) == "RETRAIN"


def test_decision_engine_monitor_on_high_increasing_error():
    engine = DecisionEngine(error_threshold=40)
    assert engine.decide(drift=False, rolling_avg=45, trend=True) == "MONITOR"


def test_decision_engine_stable_otherwise():
    engine = DecisionEngine(error_threshold=40)
    assert engine.decide(drift=False, rolling_avg=None, trend=False) == "STABLE"
