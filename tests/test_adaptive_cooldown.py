from decision.adaptive_cooldown import AdaptiveCooldown


def test_high_drift_uses_min_cooldown():
    cd = AdaptiveCooldown(min_cooldown=10, max_cooldown=50)
    should, required, _ = cd.should_retrain(current_idx=10, drift_score=0.9)
    assert required == 10
    assert should is True


def test_moderate_drift_uses_mid_cooldown():
    cd = AdaptiveCooldown(min_cooldown=10, max_cooldown=50)
    should, required, _ = cd.should_retrain(current_idx=30, drift_score=0.6)
    assert required == 30
    assert should is True


def test_low_drift_uses_max_cooldown():
    cd = AdaptiveCooldown(min_cooldown=10, max_cooldown=50)
    # After a retrain at index 0, only 30 cycles have passed — should NOT retrain yet
    cd.mark_retrain(current_idx=0)
    should, required, elapsed = cd.should_retrain(current_idx=30, drift_score=0.1)
    assert required == 50
    assert elapsed == 30
    assert should is False


def test_mark_retrain_resets_cooldown():
    cd = AdaptiveCooldown(min_cooldown=10, max_cooldown=50)
    cd.mark_retrain(current_idx=100)
    should, _, elapsed = cd.should_retrain(current_idx=105, drift_score=0.9)
    assert elapsed == 5
    assert should is False


def test_get_status_returns_correct_severity():
    cd = AdaptiveCooldown(min_cooldown=10, max_cooldown=50)
    status = cd.get_status(current_idx=0, drift_score=0.9)
    assert status["drift_severity"] == "high"

    status = cd.get_status(current_idx=0, drift_score=0.6)
    assert status["drift_severity"] == "medium"

    status = cd.get_status(current_idx=0, drift_score=0.2)
    assert status["drift_severity"] == "low"
