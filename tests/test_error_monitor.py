from drift.error_monitor import ErrorMonitor


def test_error_monitor_warmup_and_trend():
    monitor = ErrorMonitor(window_size=3)

    assert monitor.update(1.0) is None
    assert monitor.update(2.0) is None

    avg = monitor.update(3.0)
    assert avg == 2.0
    assert monitor.is_increasing() is True


def test_error_monitor_not_increasing_when_flat_or_down():
    monitor = ErrorMonitor(window_size=3)
    monitor.update(3.0)
    monitor.update(2.0)
    monitor.update(1.0)

    assert monitor.is_increasing() is False
