from governance.audit_log import AuditLog


def test_append_audit_prepends_to_front():
    log = AuditLog()
    log.append_audit("Event A", "reason", "action", "Healthy", "MODEL")
    log.append_audit("Event B", "reason", "action", "Warning", "DRIFT")
    entries = log.get_audit_logs()
    assert entries[0]["event"] == "Event B"
    assert entries[1]["event"] == "Event A"


def test_append_alert_stores_severity():
    log = AuditLog()
    log.append_alert("Critical", "Something broke")
    alerts = log.get_alerts()
    assert alerts[0]["severity"] == "Critical"
    assert alerts[0]["message"] == "Something broke"


def test_append_timeline_stores_state():
    log = AuditLog()
    log.append_timeline("System online", "Healthy")
    timeline = log.get_timeline()
    assert timeline[0]["event"] == "System online"
    assert timeline[0]["state"] == "Healthy"


def test_append_model_history_deduplicates():
    log = AuditLog()
    log.append_model_history(1)
    log.append_model_history(1)  # duplicate — should not be added again
    log.append_model_history(2)
    history = log.get_model_history()
    assert len(history) == 2
    assert history[0]["modelVersion"] == "v1.0.2"
    assert history[1]["modelVersion"] == "v1.0.1"


def test_ring_buffers_respect_maxlen():
    log = AuditLog(audit_maxlen=3)
    for i in range(5):
        log.append_audit(f"Event {i}", "r", "a", "Healthy", "MODEL")
    assert len(log.get_audit_logs()) == 3
