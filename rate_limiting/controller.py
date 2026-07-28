"""
rate_limiting/controller.py - Adaptive ingestion rate controller.

Manages the streaming event queue, load-shedding, and the adaptive
rate-limit policy that protects the ML worker thread from overload.

Responsibilities:
  - Maintain the inbound event queue (deque with maxlen cap)
  - Track load-shedding statistics when the queue is full
  - Compute the applied rate limit each tick based on:
      * operator-configured ceiling
      * measured hardware capacity
      * drift / retrain activity
      * current backlog depth
  - Emit audit events via the supplied AuditLog when throttling occurs
"""

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from governance.audit_log import AuditLog

STREAM_QUEUE_MAXLEN = 500


class RateLimitController:
    """Adaptive ingestion-rate controller for the streaming ML runtime."""

    def __init__(self, rate_limit: float = 14.0, worker_capacity_limit: float = 40.0):
        # Operator-configured ceiling (eps)
        self.rate_limit: float = rate_limit
        self.rate_limit_enabled: bool = True

        # Measured hardware capacity (calibrated during warm-up)
        self.worker_capacity_limit: float = worker_capacity_limit

        # Smoothed applied limit (starts at operator ceiling)
        self.applied_rate_limit: float = rate_limit

        # Human-readable state / reason surfaced to the dashboard
        self.state: str = "Nominal"
        self.reason: str = "Incoming traffic is within the configured limit"

        # Load-shedding counters
        self.load_shedding_total: int = 0
        self.load_shedding_events_since_snapshot: int = 0
        self.last_load_shedding_at: float | None = None
        self.last_load_shedding_audit_total: int = 0

        # Inbound event queue
        self.event_queue: deque = deque(maxlen=STREAM_QUEUE_MAXLEN)
        self.stream_backlog: int = 0

        # Last sample index at which a rate-control audit was written
        self._last_audit_sample: int = -1000

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def enqueue(self, event: dict, sample_index: int, audit: "AuditLog") -> None:
        """
        Push *event* onto the inbound queue.

        If the queue is already at capacity the oldest unprocessed event
        is discarded (load-shedding) and an audit entry is written at
        milestones (first drop and every 100 drops thereafter).
        """
        import time  # local import to avoid circular deps at module level

        if len(self.event_queue) >= STREAM_QUEUE_MAXLEN:
            self.event_queue.popleft()
            self.load_shedding_total += 1
            self.load_shedding_events_since_snapshot += 1
            self.last_load_shedding_at = time.monotonic()

            if (
                self.load_shedding_total == 1
                or self.load_shedding_total - self.last_load_shedding_audit_total >= 100
            ):
                self.last_load_shedding_audit_total = self.load_shedding_total
                audit.append_audit(
                    "Sensor Data Loss - Queue Overflow",
                    "Ingestion buffer at capacity - oldest unprocessed sensor readings are "
                    "being discarded to prevent backlog growth",
                    f"Total readings dropped this session: {self.load_shedding_total} | "
                    "Immediate operator review recommended",
                    "Critical",
                    "RATE_LIMIT",
                )

        self.event_queue.append(event)
        self.stream_backlog = len(self.event_queue)

    def reset_snapshot_counters(self) -> None:
        """Reset per-snapshot load-shedding counter (call after each dashboard publish)."""
        self.load_shedding_events_since_snapshot = 0

    # ------------------------------------------------------------------
    # Adaptive rate control
    # ------------------------------------------------------------------

    def update(
        self,
        elapsed: float,
        stream_rate: float,
        latest_drift: float,
        is_retraining: bool,
        sample_index: int,
        audit: "AuditLog",
    ) -> float:
        """
        Advance the adaptive rate controller by *elapsed* seconds and
        return the current applied processing limit (events per second).

        Args:
            elapsed       : Seconds since the last tick.
            stream_rate   : Current incoming stream rate (eps) – used to
                            detect whether we are falling behind.
            latest_drift  : Most recent drift score [0, 1].
            is_retraining : True while a background retrain / shadow eval runs.
            sample_index  : Current global sample counter (for audit throttling).
            audit         : AuditLog instance to write throttle events into.

        Returns:
            The smoothed applied rate limit (eps) for this tick.
        """
        if not self.rate_limit_enabled:
            self.applied_rate_limit = self.worker_capacity_limit
            self.state = "Bypassed"
            self.reason = "Rate limiting is disabled (ML worker capacity)"
            return self.applied_rate_limit

        # --- Rule 1: Hardware ceiling always applies ---
        ceiling = float(np.clip(
            min(self.rate_limit, self.worker_capacity_limit),
            1.0,
            self.worker_capacity_limit,
        ))

        # --- Rule 2: Protect when drift is detected or retraining is active ---
        drift_triggered = latest_drift >= 0.65

        if is_retraining or drift_triggered:
            target = max(1.0, ceiling * 0.60)
            state = "Protecting"
            if is_retraining:
                reason = (
                    "Ingestion rate reduced - model refresh in progress, "
                    "processing capacity temporarily reserved"
                )
            else:
                reason = (
                    "Elevated drift index detected - ingestion rate reduced "
                    "pre-emptively to maintain model accuracy"
                )

        # --- Rule 3: Drain backlog at full ceiling when system is healthy ---
        elif self.stream_backlog > 5:
            target = ceiling
            state = "Draining"
            reason = "Clearing backlog - processing at full operator limit"

        # --- Rule 4: Nominal - no backlog, no retrain ---
        else:
            target = ceiling
            state = "Nominal"
            reason = "Incoming traffic is within the configured limit"

        # Smooth transition toward target (faster drop, slower ramp-up)
        step_ratio = 0.45 if target < self.applied_rate_limit else 0.25
        max_step = max(0.5, ceiling * step_ratio * max(elapsed, 0.25))
        delta = float(np.clip(target - self.applied_rate_limit, -max_step, max_step))
        self.applied_rate_limit = float(np.clip(self.applied_rate_limit + delta, 1.0, ceiling))

        if stream_rate > self.applied_rate_limit:
            if state in {"Nominal", "Draining"}:
                state = "Throttling"
                reason = (
                    "Sensor data arrival rate exceeds processing capacity - "
                    "readings are queuing"
                )
            elif state == "Nominal":
                reason = (
                    "Sensor data arrival rate is above the current adaptive processing limit"
                )

        previous_state = self.state
        self.state = state
        self.reason = reason

        # --- Throttle / Protecting audit (rate-limited) ---
        should_audit = (
            state in {"Throttling", "Protecting"}
            and (
                previous_state != state
                or sample_index - self._last_audit_sample >= 30
            )
            and (state == "Throttling" or self.stream_backlog > 5)
            and self.stream_backlog < STREAM_QUEUE_MAXLEN
        )
        if should_audit:
            self._last_audit_sample = sample_index
            timeline_msg = (
                "Sensor ingestion throttled - queue depth increasing"
                if state == "Throttling"
                else "Ingestion rate reduced - model refresh underway"
            )
            audit.append_timeline(timeline_msg, "Warning")
            audit.append_audit(
                "Ingestion Rate Adjusted",
                reason,
                f"Processing limit: {self.applied_rate_limit:.1f} eps | "
                f"Backlog depth: {int(self.stream_backlog)} readings",
                "Warning",
                "RATE_LIMIT",
            )

        return self.applied_rate_limit

    # ------------------------------------------------------------------
    # Controls patch (called by /api/controls)
    # ------------------------------------------------------------------

    def apply_controls(
        self,
        rate_limit: float | None,
        rate_limit_enabled: bool | None,
    ) -> None:
        """Update operator-facing controls and clamp the applied limit accordingly."""
        if rate_limit is not None:
            self.rate_limit = float(rate_limit)
        if rate_limit_enabled is not None:
            self.rate_limit_enabled = bool(rate_limit_enabled)

        if self.rate_limit_enabled:
            self.applied_rate_limit = min(self.applied_rate_limit, self.rate_limit)
        else:
            self.applied_rate_limit = self.worker_capacity_limit
            self.state = "Bypassed"
            self.reason = "Rate limiting is disabled (ML worker capacity)"
