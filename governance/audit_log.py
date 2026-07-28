"""
governance/audit_log.py - Centralized in-memory audit and timeline store.

Manages four ring-buffers that power the dashboard:
  - audit_logs   : structured event trail (shown in Governance Logs panel)
  - alerts       : operator alerts with severity levels
  - timeline     : short human-readable event feed for the Self-Healing page
  - model_history: chronological record of model promotions
"""

from collections import deque
from datetime import datetime


class AuditLog:
    """
    Thread-safe-friendly (caller holds the lock) registry of all governance
    events emitted by the streaming runtime.  All buffers are bounded ring
    queues so memory usage is constant regardless of runtime duration.
    """

    def __init__(
        self,
        audit_maxlen: int = 120,
        alerts_maxlen: int = 20,
        timeline_maxlen: int = 24,
        history_maxlen: int = 30,
    ):
        self.audit_logs: deque = deque(maxlen=audit_maxlen)
        self.alerts: deque = deque(maxlen=alerts_maxlen)
        self.timeline: deque = deque(maxlen=timeline_maxlen)
        self.model_history: deque = deque(maxlen=history_maxlen)

    # ------------------------------------------------------------------
    # Internal timestamp helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now_parts() -> tuple[str, str]:
        """Return (HH:MM:SS, YYYY-MM-DD HH:MM:SS) for the current moment."""
        now = datetime.now()
        return now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def append_audit(
        self,
        event: str,
        reason: str,
        action: str,
        status: str,
        event_type: str,
    ) -> None:
        """
        Append a structured entry to the governance audit trail.

        Args:
            event      : Short event name shown in the Event column.
            reason     : Machine-readable reason string (metrics, MAE, etc.).
            action     : Human description of what the system did.
            status     : 'Healthy' | 'Warning' | 'Critical'
            event_type : Category tag - 'MODEL' | 'DRIFT' | 'CONFIDENCE' | 'RATE_LIMIT'
        """
        _, stamp = self._now_parts()
        self.audit_logs.appendleft(
            {
                "timestamp": stamp,
                "event": event,
                "reason": reason,
                "actionTaken": action,
                "status": status,
                "eventType": event_type,
            }
        )

    def append_alert(self, severity: str, message: str) -> None:
        """
        Append an operator-facing alert.

        Args:
            severity : 'Critical' | 'Warning' | 'Info'
            message  : Short human-readable description.
        """
        clock, _ = self._now_parts()
        self.alerts.appendleft(
            {"timestamp": clock, "severity": severity, "message": message}
        )

    def append_timeline(self, event: str, state: str) -> None:
        """
        Append an entry to the self-healing timeline feed.

        Args:
            event : Short description of what happened.
            state : 'Healthy' | 'Warning' | 'Critical'
        """
        clock, _ = self._now_parts()
        self.timeline.appendleft({"time": clock, "event": event, "state": state})

    def append_model_history(self, model_version: int) -> None:
        """
        Record a model promotion event if the version has actually changed.

        Args:
            model_version : Integer suffix of the version string (e.g. 2 -> v1.0.2).
        """
        clock, _ = self._now_parts()
        version_str = f"v1.0.{model_version}"
        last = self.model_history[0] if self.model_history else None
        if last is None or last["modelVersion"] != version_str:
            self.model_history.appendleft(
                {
                    "time": clock,
                    "versionIndex": model_version,
                    "modelVersion": version_str,
                }
            )

    # ------------------------------------------------------------------
    # Public read methods (return plain lists for JSON serialisation)
    # ------------------------------------------------------------------

    def get_audit_logs(self) -> list:
        return list(self.audit_logs)

    def get_alerts(self) -> list:
        return list(self.alerts)

    def get_timeline(self) -> list:
        return list(self.timeline)

    def get_model_history(self) -> list:
        return list(self.model_history)
