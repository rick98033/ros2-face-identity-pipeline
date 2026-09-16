"""Predefined Degradation Response Framework (CP-009).

A declarative schema and tracking type that requires each component to
enumerate its dependency failure modes, assign a predefined response to
each, and make degradation state observable via health endpoints and
structured events.

Design requirements:
  - SI-7.1: Predefined degradation response for every dependency failure
  - SI-9.1: Explicit, documented recovery paths

Distinct from:
  - CP-002 (WatchdogThread) — detects stalls; CP-009 defines *what to do*
  - CP-003 (FreshValue)     — detects staleness; CP-009 defines the response
  - CP-007 (StructuredEvents) — emission format; CP-009 is the decision framework
  - CP-004 (HealthEndpoints)  — exposure surface; CP-009 provides the state

Usage:
    from thor_behavior import (
        DegradationPolicy, DegradedState, DepPolicy,
        FailureMode, ResponseType,
    )

    policy = DegradationPolicy("my_component", [
        DepPolicy(
            dependency="llm_server",
            failure_modes=[FailureMode.UNAVAILABLE, FailureMode.TIMEOUT],
            response=ResponseType.DEGRADE_WITH_NOTIFICATION,
            recovery_action="Wait for LLM server to become available",
        ),
    ])
    state = DegradedState(policy)

    # When a dependency fails:
    response = state.report_degradation(
        "llm_server", FailureMode.UNAVAILABLE, "connection refused"
    )
    # response == ResponseType.DEGRADE_WITH_NOTIFICATION

    # When it recovers:
    state.report_recovery("llm_server")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from thor_telemetry import StructuredLogger


class FailureMode(Enum):
    """How a dependency can fail."""
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    DATA_STALE = "data_stale"
    CREDENTIAL_INVALID = "credential_invalid"
    RESOURCE_EXHAUSTED = "resource_exhausted"


# CP-003: Lifecycle state sets for FailureMode.
TERMINAL_STATES = frozenset({FailureMode.UNKNOWN})
ACTIVE_STATES = frozenset({
    FailureMode.UNAVAILABLE, FailureMode.TIMEOUT,
    FailureMode.DATA_STALE, FailureMode.CREDENTIAL_INVALID,
    FailureMode.RESOURCE_EXHAUSTED,
})


class ResponseType(Enum):
    """Predefined response to a dependency failure."""
    UNKNOWN = "unknown"
    FAIL_SAFE = "fail_safe"
    DEGRADE_WITH_NOTIFICATION = "degrade_with_notification"
    RETRY_WITH_BACKOFF = "retry_with_backoff"


class RecoveryTarget(Enum):
    """Declared end state after recovery from degradation (SI-9.1).

    Every dependency policy must declare what "recovered" means:
    - FULL_OPERATION: all capabilities restored, no operator action needed
    - DEGRADED_OPERATION: partial service, some capabilities unavailable
    - SAFE_STOPPED: component halted safely, requires manual restart
    """
    UNKNOWN = "unknown"
    FULL_OPERATION = "full_operation"
    DEGRADED_OPERATION = "degraded_operation"
    SAFE_STOPPED = "safe_stopped"


class UndeclaredDependencyError(Exception):
    """Raised when report_degradation is called for a dependency not in the policy."""

    def __init__(self, dependency: str) -> None:
        super().__init__(
            f"Dependency {dependency!r} was not declared in the DegradationPolicy. "
            f"Every dependency must be declared upfront (SI-7.1)."
        )
        self.dependency = dependency


@dataclass(frozen=True)
class DepPolicy:
    """Declaration of a single dependency's failure modes and response.

    Args:
        dependency: Name of the upstream dependency.
        failure_modes: How the dependency can fail.
        response: Predefined response to any failure of this dependency.
        recovery_action: Human-readable description of how to recover.
        recovery_target: Declared end state after recovery (SI-9.1).
        max_retries: For RETRY_WITH_BACKOFF only. None = unlimited with
            capped backoff.
        backoff_base_sec: For RETRY_WITH_BACKOFF only. Base interval for
            exponential backoff.
    """
    dependency: str
    failure_modes: list[FailureMode]
    response: ResponseType
    recovery_action: str
    recovery_target: RecoveryTarget = RecoveryTarget.FULL_OPERATION
    max_retries: Optional[int] = None
    backoff_base_sec: Optional[float] = None


@dataclass
class DegradationRecord:
    """Runtime record of a currently degraded dependency."""
    dependency: str
    failure_mode: FailureMode
    response: ResponseType
    detail: str
    recovery_action: str
    recovery_target: RecoveryTarget
    since: float  # time.monotonic() when degradation was first reported
    last_event_time: float  # time.monotonic() of last emitted event


class DegradationPolicy:
    """Static declaration of a component's dependency failure modes.

    Immutable after construction. Each dependency must be unique.
    RETRY_WITH_BACKOFF policies require backoff_base_sec > 0.

    Args:
        component: Name of the component owning this policy.
        policies: List of per-dependency policy declarations.

    Raises:
        ValueError: If policies is empty, has duplicate dependencies,
            or RETRY_WITH_BACKOFF is used without valid backoff_base_sec.
    """

    def __init__(self, component: str, policies: list[DepPolicy]) -> None:
        if not policies:
            raise ValueError("At least one DepPolicy is required")

        seen: set[str] = set()
        for p in policies:
            if p.dependency in seen:
                raise ValueError(
                    f"Duplicate dependency {p.dependency!r} in policies"
                )
            seen.add(p.dependency)
            if p.response == ResponseType.RETRY_WITH_BACKOFF:
                if p.backoff_base_sec is None or p.backoff_base_sec <= 0:
                    raise ValueError(
                        f"RETRY_WITH_BACKOFF for {p.dependency!r} requires "
                        f"backoff_base_sec > 0, got {p.backoff_base_sec!r}"
                    )

        self._component = component
        self._policies: dict[str, DepPolicy] = {p.dependency: p for p in policies}

    @property
    def component(self) -> str:
        return self._component

    def get(self, dependency: str) -> Optional[DepPolicy]:
        """Look up policy for a dependency, or None if not declared."""
        return self._policies.get(dependency)

    @property
    def dependencies(self) -> frozenset[str]:
        return frozenset(self._policies.keys())

    def __len__(self) -> int:
        return len(self._policies)


class DegradedState:
    """Runtime tracker of degradation state for a component.

    Thread-safe. Reports degradation events and recoveries, emits
    structured log events (with debounce), and exposes health counters
    compatible with CP-004 health endpoint convention.

    Args:
        policy: The component's degradation policy (immutable).
        logger: Optional StructuredLogger. Falls back to standard logging.
        debounce_sec: Minimum interval between repeated degradation_started
            events for the same dependency. Construction-time only, immutable.
            Controls log volume, not state tracking or counters.
    """

    def __init__(
        self,
        policy: DegradationPolicy,
        logger: Optional[StructuredLogger] = None,
        debounce_sec: float = 30.0,
    ) -> None:
        self._policy = policy
        self._slog = logger
        self._flog = logging.getLogger(f"degradation.{policy.component}")
        self._debounce_sec = debounce_sec

        self._lock = threading.Lock()
        self._records: dict[str, DegradationRecord] = {}
        self._degradation_events_total = 0
        self._recovery_events_total = 0

    @property
    def debounce_sec(self) -> float:
        return self._debounce_sec

    def report_degradation(
        self,
        dependency: str,
        failure_mode: FailureMode,
        detail: str,
    ) -> ResponseType:
        """Report a dependency failure. Returns the predefined response.

        Always increments degradation_events_total. Emits a structured
        event subject to debounce window (repeated reports within
        debounce_sec for the same dependency suppress the event but
        still update detail/timestamp and increment the counter).

        Raises:
            UndeclaredDependencyError: If dependency is not in the policy.
        """
        dep_policy = self._policy.get(dependency)
        if dep_policy is None:
            raise UndeclaredDependencyError(dependency)

        now = time.monotonic()
        should_emit = False

        with self._lock:
            self._degradation_events_total += 1
            existing = self._records.get(dependency)

            if existing is None:
                # New degradation
                self._records[dependency] = DegradationRecord(
                    dependency=dependency,
                    failure_mode=failure_mode,
                    response=dep_policy.response,
                    detail=detail,
                    recovery_action=dep_policy.recovery_action,
                    recovery_target=dep_policy.recovery_target,
                    since=now,
                    last_event_time=now,
                )
                should_emit = True
            else:
                # Already degraded — update detail, check debounce
                existing.failure_mode = failure_mode
                existing.detail = detail
                if (now - existing.last_event_time) >= self._debounce_sec:
                    existing.last_event_time = now
                    should_emit = True

        if should_emit:
            self._emit_degradation_started(
                dependency, failure_mode, dep_policy.response, detail
            )

        return dep_policy.response

    def report_recovery(self, dependency: str) -> None:
        """Report that a dependency has recovered.

        Emits a degradation_recovered event with duration. No-op if
        the dependency is not currently degraded.
        """
        now = time.monotonic()
        duration_sec: Optional[float] = None

        with self._lock:
            existing = self._records.pop(dependency, None)
            if existing is not None:
                self._recovery_events_total += 1
                duration_sec = now - existing.since

        if duration_sec is not None:
            self._emit_degradation_recovered(dependency, duration_sec)

    def is_degraded(self) -> bool:
        """True if any dependency is currently degraded."""
        with self._lock:
            return len(self._records) > 0

    def get_degradations(self) -> dict[str, dict]:
        """Current degradation state per dependency.

        Returns a dict keyed by dependency name, compatible with
        CP-004 health endpoint response schema.
        """
        now = time.monotonic()
        with self._lock:
            return {
                dep: {
                    "dependency": rec.dependency,
                    "failure_mode": rec.failure_mode.value,
                    "response": rec.response.value,
                    "detail": rec.detail,
                    "recovery_action": rec.recovery_action,
                    "recovery_target": rec.recovery_target.value,
                    "since": rec.since,
                    "duration_sec": round(now - rec.since, 3),
                }
                for dep, rec in self._records.items()
            }

    def get_health(self) -> dict:
        """Health counters compatible with CP-004 health endpoint convention."""
        with self._lock:
            return {
                "component": self._policy.component,
                "degradation_events_total": self._degradation_events_total,
                "recovery_events_total": self._recovery_events_total,
                "currently_degraded_count": len(self._records),
                "dependencies_declared": len(self._policy),
            }

    # ── Internal event emission ──

    def _emit_degradation_started(
        self,
        dependency: str,
        failure_mode: FailureMode,
        response: ResponseType,
        detail: str,
    ) -> None:
        fields = dict(
            component=self._policy.component,
            dependency=dependency,
            failure_mode=failure_mode.value,
            response_type=response.value,
            detail=detail,
        )
        if self._slog:
            self._slog.warning("degradation_started", **fields)
        else:
            self._flog.warning("degradation_started: %s", fields)

    def _emit_degradation_recovered(
        self, dependency: str, duration_sec: float
    ) -> None:
        fields = dict(
            component=self._policy.component,
            dependency=dependency,
            duration_sec=round(duration_sec, 3),
        )
        if self._slog:
            self._slog.info("degradation_recovered", **fields)
        else:
            self._flog.info("degradation_recovered: %s", fields)
