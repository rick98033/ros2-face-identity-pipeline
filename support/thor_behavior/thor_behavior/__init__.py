"""Shared behavior coordinator infrastructure for Thor coordinators.

Provides the generalized state model, configuration, and base class
that all behavior coordinators (follow-me, dock, go-to-room, etc.)
must subclass.

Usage:
    from thor_behavior import BehaviorCoordinatorBase, BehaviorConfig, BehaviorState

    config = BehaviorConfig(behavior_type="dock", ...)
    class DockCoordinator(BehaviorCoordinatorBase):
        ...

This curated copy is retained because the ROS nodes use its watchdog and
degradation primitives. The full behavior-coordinator reference lives in the
companion safe-follow-me-reference repository.
"""

from .core import (
    BehaviorState,
    TERMINAL_STATES,
    ACTIVE_STATES,
    BehaviorConfig,
    BehaviorCoordinatorBase,
    ComponentHealth,
    ReadinessSnapshot,
    ErrorInfo,
    LastSessionInfo,
    HealthCounters,
    StopRequest,
    HealthEvent,
    BudgetExpired,
)
from .milestones import (
    MilestoneBase,
    MilestoneCallback,
    CommandAccepted,
    PrereqCheckStarted,
    PrereqCheckResult,
    PhaseChanged,
    Progress,
    RecoverableIssue,
    Fault,
    StopEscalationLevel,
    TerminalOutcome,
)
from .state_reader import StateReader
from .watchdog import WatchdogState, WatchdogThread
from .degradation import (
    DegradationPolicy,
    DegradedState,
    DegradationRecord,
    DepPolicy,
    FailureMode,
    RecoveryTarget,
    ResponseType,
    UndeclaredDependencyError,
)

__all__ = [
    "BehaviorState",
    "TERMINAL_STATES",
    "ACTIVE_STATES",
    "BehaviorConfig",
    "BehaviorCoordinatorBase",
    "ComponentHealth",
    "ReadinessSnapshot",
    "ErrorInfo",
    "LastSessionInfo",
    "HealthCounters",
    "StopRequest",
    "HealthEvent",
    "BudgetExpired",
    "MilestoneBase",
    "MilestoneCallback",
    "CommandAccepted",
    "PrereqCheckStarted",
    "PrereqCheckResult",
    "PhaseChanged",
    "Progress",
    "RecoverableIssue",
    "Fault",
    "StopEscalationLevel",
    "TerminalOutcome",
    "StateReader",
    "WatchdogState",
    "WatchdogThread",
    "DegradationPolicy",
    "DegradedState",
    "DegradationRecord",
    "DepPolicy",
    "FailureMode",
    "RecoveryTarget",
    "ResponseType",
    "UndeclaredDependencyError",
]
