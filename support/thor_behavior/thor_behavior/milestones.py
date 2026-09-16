"""Milestone event types for behavior coordinators.

Frozen dataclasses emitted by BehaviorCoordinatorBase at key lifecycle points.
No ROS dependency — pure Python. Consumed by VoiceOrchestrator and logging.

All milestones share MilestoneBase fields:
- session_id: active session
- behavior_type: "follow_me", "dock", etc.
- monotonic_ts: time.monotonic() — ordering within process
- wall_time_ms: int(time.time() * 1000) — cross-process correlation
- sequence: per-session monotonic counter
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class MilestoneBase:
    """Base fields shared by all milestone types."""
    session_id: str
    behavior_type: str
    monotonic_ts: float      # time.monotonic()
    wall_time_ms: int        # int(time.time() * 1000)
    sequence: int            # per-session monotonic counter


@dataclass(frozen=True)
class CommandAccepted(MilestoneBase):
    """Start request passed gates, session is beginning."""
    pass


@dataclass(frozen=True)
class PrereqCheckStarted(MilestoneBase):
    """Readiness check initiated for a component."""
    component: str = ""


@dataclass(frozen=True)
class PrereqCheckResult(MilestoneBase):
    """Readiness check completed for a component."""
    component: str = ""
    passed: bool = True
    reason_code: str = ""


@dataclass(frozen=True)
class PhaseChanged(MilestoneBase):
    """State transition committed. Emitted AFTER state write (MS-1)."""
    from_state: str = ""
    to_state: str = ""
    alias: str = ""       # display alias (e.g., "ARMED" for STARTING)
    reason: str = ""


@dataclass(frozen=True)
class Progress(MilestoneBase):
    """Behavior-specific progress update."""
    percent: float | None = None
    step: str | None = None


@dataclass(frozen=True)
class RecoverableIssue(MilestoneBase):
    """Soft fault detected — may self-resolve within grace budget."""
    issue_code: str = ""
    suggested_user_action: str = ""


@dataclass(frozen=True)
class Fault(MilestoneBase):
    """Hard fault or escalated soft fault — unrecoverable."""
    error_code: str = ""
    severity: str = "CRITICAL"     # CRITICAL or WARNING
    recoverable: bool = False


@dataclass(frozen=True)
class StopEscalationLevel(MilestoneBase):
    """Stop escalation level attempted."""
    level_name: str = ""
    outcome: str = ""          # ok, timeout, error, rejected
    elapsed_ms: int = 0


@dataclass(frozen=True)
class TerminalOutcome(MilestoneBase):
    """Session ended — exactly one per session terminal state (MS-3)."""
    outcome: str = ""          # FAILED, CANCELLED, SUCCEEDED
    reason_code: str = ""
    error_detail: str = ""


# Callback type: synchronous, must not raise (wrapped in try/except)
MilestoneCallback = Callable[[MilestoneBase], None]
