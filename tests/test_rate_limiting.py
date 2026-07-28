import pandas as pd
import numpy as np
from governance.audit_log import AuditLog
from rate_limiting.controller import RateLimitController, STREAM_QUEUE_MAXLEN


def _make_audit():
    return AuditLog()


def test_enqueue_adds_events():
    rl = RateLimitController()
    audit = _make_audit()
    rl.enqueue({"data": "a"}, sample_index=0, audit=audit)
    rl.enqueue({"data": "b"}, sample_index=1, audit=audit)
    assert len(rl.event_queue) == 2


def test_enqueue_load_sheds_at_capacity():
    rl = RateLimitController()
    audit = _make_audit()
    for i in range(STREAM_QUEUE_MAXLEN + 5):
        rl.enqueue({"i": i}, sample_index=i, audit=audit)
    assert len(rl.event_queue) == STREAM_QUEUE_MAXLEN
    assert rl.load_shedding_total == 5


def test_update_returns_float_rate():
    rl = RateLimitController(rate_limit=10.0, worker_capacity_limit=40.0)
    audit = _make_audit()
    rate = rl.update(
        elapsed=0.1,
        stream_rate=5.0,
        latest_drift=0.0,
        is_retraining=False,
        sample_index=0,
        audit=audit,
    )
    assert isinstance(rate, float)
    assert 1.0 <= rate <= 40.0


def test_rate_limit_disabled_bypasses_ceiling():
    rl = RateLimitController(rate_limit=5.0, worker_capacity_limit=40.0)
    audit = _make_audit()
    rl.apply_controls(rate_limit=None, rate_limit_enabled=False)
    rate = rl.update(
        elapsed=0.1,
        stream_rate=5.0,
        latest_drift=0.0,
        is_retraining=False,
        sample_index=0,
        audit=audit,
    )
    assert rl.state == "Bypassed"
    assert rate == rl.worker_capacity_limit


def test_apply_controls_updates_rate_limit():
    rl = RateLimitController(rate_limit=14.0)
    rl.apply_controls(rate_limit=20.0, rate_limit_enabled=True)
    assert rl.rate_limit == 20.0


def test_reset_snapshot_counters():
    rl = RateLimitController()
    audit = _make_audit()
    for i in range(STREAM_QUEUE_MAXLEN + 10):
        rl.enqueue({"i": i}, sample_index=i, audit=audit)
    rl.reset_snapshot_counters()
    assert rl.load_shedding_events_since_snapshot == 0
