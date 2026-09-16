"""Cross-boundary state reconciliation tracker (CP-005, SI-6.2).

Tracks per-consumer reconciliation state for cross-boundary lifecycles.
Three states: SYNCED, RECONCILING, UNREACHABLE. Thread-safe via Lock.

Usage:
    from thor_telemetry import ReconciliationTracker, ReconciliationState

    tracker = ReconciliationTracker(
        lifecycle="follow_me",
        consumer="witness/follow-me-coordinator",
        reconciliation_timeout_sec=5.0,
        unreachable_policy="fail_safe",
    )

    # Authority pushed new state
    result = tracker.on_authority_state("RUNNING", local_state="RUNNING")
    # result.diverged is False — states match

    # On reconnect, begin reconciliation
    tracker.begin_reconciliation()
    # ... query authority ...
    result = tracker.on_authority_response("FAILED", local_state="RUNNING")
    # result.diverged is True, result.resolution == "adopted"
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass
from typing import Optional


class ReconciliationState(str, enum.Enum):
    """Per-consumer reconciliation state (CP-005 §3)."""
    UNKNOWN = "UNKNOWN"
    SYNCED = "SYNCED"
    RECONCILING = "RECONCILING"
    UNREACHABLE = "UNREACHABLE"


# CP-003: Lifecycle state sets for ReconciliationState.
TERMINAL_STATES = frozenset({ReconciliationState.UNREACHABLE, ReconciliationState.UNKNOWN})
ACTIVE_STATES = frozenset({ReconciliationState.RECONCILING})


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of a reconciliation check."""
    diverged: bool
    prior_local_state: str
    authority_state: str
    resolution: str  # "none" | "adopted" | "safety_override"
    staleness_sec: float


@dataclass(frozen=True)
class ReconciliationHealth:
    """Health counters for CP-004 exposure."""
    reconciliation_attempts_total: int
    reconciliation_failures_total: int
    divergences_resolved_total: int
    safety_overrides_total: int
    current_state: str  # SYNCED | RECONCILING | UNREACHABLE


class ReconciliationTracker:
    """Tracks reconciliation state for one consumer of one lifecycle.

    Thread-safe: all mutations protected by a Lock. Read-only health
    snapshot via get_health() is also lock-protected.

    Unreachable policies:
    - "fail_safe": caller must transition to safe/inactive state immediately
    - "hold_with_staleness": caller may hold last known state for max_hold_sec,
      after which fail_safe applies

    Per convention §3, initial state is RECONCILING — consumer must query
    authority before presenting lifecycle state (startup recovery).
    """

    def __init__(
        self,
        lifecycle: str,
        consumer: str,
        reconciliation_timeout_sec: float = 5.0,
        unreachable_policy: str = "fail_safe",
        max_hold_sec: Optional[float] = None,
        safety_relevant: bool = False,
    ) -> None:
        if unreachable_policy not in ("fail_safe", "hold_with_staleness"):
            raise ValueError(f"Invalid unreachable_policy: {unreachable_policy}")
        if unreachable_policy == "hold_with_staleness" and max_hold_sec is None:
            raise ValueError("max_hold_sec required for hold_with_staleness policy")

        self._lifecycle = lifecycle
        self._consumer = consumer
        self._reconciliation_timeout_sec = reconciliation_timeout_sec
        self._unreachable_policy = unreachable_policy
        self._max_hold_sec = max_hold_sec
        self._safety_relevant = safety_relevant

        self._lock = threading.Lock()
        # Initial state: RECONCILING per convention §3 (startup recovery)
        self._state = ReconciliationState.RECONCILING
        self._last_synced_time: float = 0.0
        self._reconciling_since: float = time.monotonic()
        self._unreachable_since: float = 0.0
        self._last_authority_state: str = ""

        # Monotonic counters (CP-004, SI-6.3)
        self._attempts_total: int = 0
        self._failures_total: int = 0
        self._divergences_total: int = 0
        self._safety_overrides_total: int = 0

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def consumer(self) -> str:
        return self._consumer

    @property
    def state(self) -> ReconciliationState:
        with self._lock:
            return self._state

    def on_authority_state(
        self,
        authority_state: str,
        local_state: str,
    ) -> ReconciliationResult:
        """Authority pushed a new state. Check for divergence.

        Called on every authority event (on_authority_event trigger).
        """
        with self._lock:
            self._attempts_total += 1
            now = time.monotonic()
            staleness = now - self._last_synced_time if self._last_synced_time > 0 else 0.0
            self._last_authority_state = authority_state

            if authority_state == local_state:
                # No divergence — mark synced
                self._state = ReconciliationState.SYNCED
                self._last_synced_time = now
                return ReconciliationResult(
                    diverged=False,
                    prior_local_state=local_state,
                    authority_state=authority_state,
                    resolution="none",
                    staleness_sec=staleness,
                )

            # Divergence detected
            self._divergences_total += 1

            # Safety override: consumer keeps more-restrictive state
            if self._safety_relevant and self._is_more_restrictive(local_state, authority_state):
                self._safety_overrides_total += 1
                self._state = ReconciliationState.SYNCED
                self._last_synced_time = now
                return ReconciliationResult(
                    diverged=True,
                    prior_local_state=local_state,
                    authority_state=authority_state,
                    resolution="safety_override",
                    staleness_sec=staleness,
                )

            # Normal resolution: authority wins
            self._state = ReconciliationState.SYNCED
            self._last_synced_time = now
            return ReconciliationResult(
                diverged=True,
                prior_local_state=local_state,
                authority_state=authority_state,
                resolution="adopted",
                staleness_sec=staleness,
            )

    def on_authority_response(
        self,
        authority_state: str,
        local_state: str,
    ) -> ReconciliationResult:
        """Authority responded to a reconciliation query.

        Called after begin_reconciliation() when the query succeeds.
        """
        with self._lock:
            self._attempts_total += 1
            now = time.monotonic()
            staleness = now - self._last_synced_time if self._last_synced_time > 0 else 0.0
            self._last_authority_state = authority_state

            if authority_state == local_state:
                self._state = ReconciliationState.SYNCED
                self._last_synced_time = now
                return ReconciliationResult(
                    diverged=False,
                    prior_local_state=local_state,
                    authority_state=authority_state,
                    resolution="none",
                    staleness_sec=staleness,
                )

            self._divergences_total += 1

            if self._safety_relevant and self._is_more_restrictive(local_state, authority_state):
                self._safety_overrides_total += 1
                self._state = ReconciliationState.SYNCED
                self._last_synced_time = now
                return ReconciliationResult(
                    diverged=True,
                    prior_local_state=local_state,
                    authority_state=authority_state,
                    resolution="safety_override",
                    staleness_sec=staleness,
                )

            self._state = ReconciliationState.SYNCED
            self._last_synced_time = now
            return ReconciliationResult(
                diverged=True,
                prior_local_state=local_state,
                authority_state=authority_state,
                resolution="adopted",
                staleness_sec=staleness,
            )

    def begin_reconciliation(self) -> None:
        """Enter RECONCILING state (e.g., on reconnect trigger)."""
        with self._lock:
            if self._state != ReconciliationState.RECONCILING:
                self._state = ReconciliationState.RECONCILING
                self._reconciling_since = time.monotonic()

    def on_authority_unreachable(self) -> None:
        """Authority could not be reached. Enter UNREACHABLE."""
        with self._lock:
            self._failures_total += 1
            self._state = ReconciliationState.UNREACHABLE
            self._unreachable_since = time.monotonic()

    def check_timeout(self) -> Optional[str]:
        """Check for reconciliation timeout or hold expiry.

        Returns:
            None: no action needed
            "fail_safe": caller must transition to safe state
            "hold_expired": hold_with_staleness max_hold_sec exceeded
        """
        with self._lock:
            now = time.monotonic()

            if self._state == ReconciliationState.RECONCILING:
                if self._reconciling_since > 0:
                    elapsed = now - self._reconciling_since
                    if elapsed > self._reconciliation_timeout_sec:
                        self._failures_total += 1
                        self._state = ReconciliationState.UNREACHABLE
                        self._unreachable_since = now
                        if self._unreachable_policy == "fail_safe":
                            return "fail_safe"
                        return None  # hold_with_staleness starts now

            if self._state == ReconciliationState.UNREACHABLE:
                if self._unreachable_policy == "fail_safe":
                    return "fail_safe"
                if (self._unreachable_policy == "hold_with_staleness"
                        and self._max_hold_sec is not None
                        and self._unreachable_since > 0):
                    elapsed = now - self._unreachable_since
                    if elapsed > self._max_hold_sec:
                        return "hold_expired"

            return None

    def reset(self) -> None:
        """Reset to SYNCED when parent lifecycle ends."""
        with self._lock:
            self._state = ReconciliationState.SYNCED
            self._last_synced_time = time.monotonic()
            self._reconciling_since = 0.0
            self._unreachable_since = 0.0

    def get_health(self) -> ReconciliationHealth:
        """Get health counters snapshot (CP-004, SI-6.3)."""
        with self._lock:
            return ReconciliationHealth(
                reconciliation_attempts_total=self._attempts_total,
                reconciliation_failures_total=self._failures_total,
                divergences_resolved_total=self._divergences_total,
                safety_overrides_total=self._safety_overrides_total,
                current_state=self._state.value,
            )

    def as_health_dict(self) -> dict:
        """Get health counters as a plain dict for JSON serialization."""
        h = self.get_health()
        return {
            "lifecycle": self._lifecycle,
            "consumer": self._consumer,
            "reconciliation_attempts_total": h.reconciliation_attempts_total,
            "reconciliation_failures_total": h.reconciliation_failures_total,
            "divergences_resolved_total": h.divergences_resolved_total,
            "safety_overrides_total": h.safety_overrides_total,
            "current_reconciliation_state": h.current_state,
        }

    @staticmethod
    def _is_more_restrictive(local_state: str, authority_state: str) -> bool:
        """Determine if local_state is more restrictive than authority_state.

        Restrictiveness order (most → least):
        FAULTED > SAFE_HOLD > STOPPING > FAILED > CANCELLED > PAUSED > IDLE > STARTING > RUNNING

        Returns True if local is more restrictive (should be kept per safety override).
        """
        _RESTRICTIVENESS = {
            "FAULTED": 100,
            "SAFE_HOLD": 90,
            "STOPPING": 80,
            "FAILED": 70,
            "CANCELLED": 60,
            "PAUSED": 50,
            "IDLE": 30,
            "STARTING": 20,
            "RUNNING": 10,
            "FOLLOWING": 10,
            "ARMED": 20,
        }
        local_r = _RESTRICTIVENESS.get(local_state.upper(), 0)
        authority_r = _RESTRICTIVENESS.get(authority_state.upper(), 0)
        return local_r > authority_r
