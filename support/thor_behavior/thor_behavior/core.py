"""Behavior coordinator base class — generalized state model for all robot behaviors.

Extracts the generic coordinator pattern from follow_me_coordinator.py into a
reusable ABC. Subclasses implement 6 abstract methods for behavior-specific
logic; the base class owns the event loop, state transitions, budget timers,
health monitoring, stop escalation, and readiness evaluation.

Concurrency model:
- All state mutations flow through a serialized asyncio.Queue
- State reads use a threading.Lock-protected snapshot
- ROS callbacks enqueue via _enqueue_threadsafe (non-blocking)
- Budget timers enqueue BudgetExpired events after async sleep

The full behavior-coordinator specifications and invariant tests live in the
companion safe-follow-me-reference repository.
"""

from __future__ import annotations

import abc
import asyncio
import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from thor_telemetry import StructuredLogger

from thor_telemetry import get_logger, FreshValue

from .milestones import (
    MilestoneBase,
    MilestoneCallback,
    CommandAccepted,
    PhaseChanged,
    Fault,
    StopEscalationLevel,
    TerminalOutcome,
)
from .watchdog import WatchdogThread


# =============================================================================
# State Model
# =============================================================================


class BehaviorState(str, enum.Enum):
    """Generalized minimum state model — all behaviors MUST implement these."""
    UNKNOWN = "UNKNOWN"
    IDLE = "IDLE"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset({
    BehaviorState.UNKNOWN,
    BehaviorState.SUCCEEDED,
    BehaviorState.FAILED,
    BehaviorState.CANCELLED,
})

ACTIVE_STATES = frozenset({
    BehaviorState.STARTING,
    BehaviorState.RUNNING,
    BehaviorState.PAUSED,
})


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class ComponentHealth:
    """Health status of a single monitored component."""
    status: str  # OK / DEGRADED / ERROR / MISSING
    detail: str
    last_seen: float  # monotonic timestamp


@dataclass
class ReadinessSnapshot:
    """Aggregated readiness from all monitored components."""
    overall: str  # READY / DEGRADED / UNAVAILABLE
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    blocking_issues: list[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class ErrorInfo:
    """Latched error information for FAILED state."""
    code: str
    detail: str
    timestamp: float


@dataclass
class LastSessionInfo:
    """Latched info about the most recent completed session."""
    state: str  # Terminal state (FAILED / CANCELLED / SUCCEEDED)
    error_code: str
    error_detail: str
    ended_at: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "ended_at": self.ended_at,
            "reason": self.reason,
        }


@dataclass
class HealthCounters:
    """Monotonic counters since process start. Exposed via GET /<behavior>/health."""
    uptime_start: float
    sessions_started: int = 0
    sessions_succeeded: int = 0
    sessions_failed: int = 0
    sessions_cancelled: int = 0
    pauses: int = 0
    stop_requests: int = 0
    stop_escalations: int = 0
    event_queue_drops: int = 0
    exceptions_in_event_loop: int = 0
    unknown_events: int = 0
    health_http_failures: int = 0


# =============================================================================
# Event Types
# =============================================================================


@dataclass
class _StartRequest:
    """Internal start event. Created by request_start(), handled by base."""
    session_id: str
    turn_id: str
    params: dict[str, Any]
    trace_ctx: Any | None = None
    result_future: asyncio.Future | None = None


@dataclass
class StopRequest:
    """Stop event. Created by request_stop(), handled by base."""
    session_id: str
    turn_id: str
    trace_ctx: Any | None = None
    result_future: asyncio.Future | None = None


@dataclass
class HealthEvent:
    """Health status update from background monitor or ROS callback."""
    component: str
    status: str  # OK / ERROR / DEGRADED / MISSING
    detail: str
    timestamp: float


@dataclass
class BudgetExpired:
    """Timer expiry event. Base handles known names; unknown forwarded to subclass."""
    which: str


# Standard budget timer names handled by the base class
_BASE_BUDGET_NAMES = frozenset({
    "starting", "running", "grace",
    "failed_linger", "cancelled_linger", "succeeded_linger",
})

# Hard ceiling margin for stop escalation (CP-008)
# ceiling = sum(level_timeouts) + margin
# Margin covers inter-level overhead (3 transitions × 100ms for asyncio scheduling,
# gRPC call setup, structured logging) + safety margin (Python GIL contention,
# Jetson thermal throttling, asyncio event loop delays, GC pauses).
# Total: 300ms overhead + 600ms safety = 900ms.
_STOP_CEILING_MARGIN_SEC = 0.9

# CP-016: Declared stop escalation budget for _handle_base_stop().
# Default ceiling = cancel(0.5) + velocity_halt(0.3) + safe_hold(0.3) + margin(0.9).
# Actual ceiling is derived dynamically in _stop_escalation() from BehaviorConfig
# level timeouts + _STOP_CEILING_MARGIN_SEC; this constant documents the default.
_STOP_ESCALATION_BUDGET_SEC = 2.0


class StopOutcome(enum.Enum):
    """Outcome of a stop escalation sequence (CP-016)."""
    UNKNOWN = "unknown"        # Uninitialized / unrecognized (CP-010)
    CONFIRMED = "confirmed"    # All levels completed successfully
    TIMEOUT = "timeout"        # Ceiling exceeded
    ERROR = "error"            # Exception during escalation
    REJECTED = "rejected"      # Stop not applicable in current state


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class BehaviorConfig:
    """Immutable configuration for a behavior coordinator.

    Budget defaults can be overridden via constructor or YAML.
    Hard ceiling fields (*_max_sec) are safety limits — never override.
    """
    behavior_type: str  # "follow_me", "dock", "go_to_room"

    # Queue
    event_queue_size: int = 256

    # Starting budget (STARTING state timeout)
    starting_budget_sec: float = 8.0
    starting_budget_max_sec: float = 15.0  # HARD CEILING

    # Running budget (RUNNING session timeout)
    running_budget_default_sec: float = 30.0
    running_budget_max_sec: float = 120.0  # HARD CEILING

    # Grace budget (PAUSED timeout)
    grace_budget_sec: float = 3.0
    grace_budget_max_sec: float = 5.0  # HARD CEILING

    # Stop escalation — per-level timeouts (authoritative values, CP-008)
    # Level budgets account for confirmed-receipt round-trip latency across
    # Docker bridge network + base controller processing time.
    cancel_timeout_sec: float = 0.5          # level (a)
    velocity_halt_timeout_sec: float = 0.3   # level (b) — 300ms for confirmed stop
    safe_hold_timeout_sec: float = 0.3       # level (c) — 300ms for confirmed mode change

    # Linger (terminal -> IDLE auto-clear)
    failed_linger_sec: float = 5.0
    cancelled_linger_sec: float = 2.0
    succeeded_linger_sec: float = 2.0

    # Health monitoring
    health_poll_interval_sec: float = 0.5
    health_http_timeout_sec: float = 1.0
    health_endpoints: dict[str, str] = field(default_factory=dict)
    health_endpoint_blocking: frozenset[str] = field(default_factory=frozenset)
    health_staleness_sec: float = 5.0

    # State display aliases (for backward-compat API responses)
    # e.g. {BehaviorState.STARTING: "ARMED", BehaviorState.RUNNING: "FOLLOWING"}
    state_aliases: dict[BehaviorState, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.starting_budget_sec > self.starting_budget_max_sec:
            raise ValueError(
                f"starting_budget_sec ({self.starting_budget_sec}) > "
                f"starting_budget_max_sec ({self.starting_budget_max_sec})"
            )
        if self.running_budget_default_sec > self.running_budget_max_sec:
            raise ValueError(
                f"running_budget_default_sec ({self.running_budget_default_sec}) > "
                f"running_budget_max_sec ({self.running_budget_max_sec})"
            )
        if self.grace_budget_sec > self.grace_budget_max_sec:
            raise ValueError(
                f"grace_budget_sec ({self.grace_budget_sec}) > "
                f"grace_budget_max_sec ({self.grace_budget_max_sec})"
            )


# =============================================================================
# BehaviorCoordinatorBase
# =============================================================================


class BehaviorCoordinatorBase(abc.ABC):
    """Abstract base class for all behavior coordinators.

    Owns the event loop, state machine, budget timers, health monitoring,
    stop escalation, and readiness evaluation. Subclasses implement 6
    abstract methods for behavior-specific logic.

    Thread safety:
    - All mutations through _event_queue (asyncio.Queue, serialized)
    - State reads through _snapshot_lock (threading.Lock)
    - ROS callbacks never block (queue.put_nowait via call_soon_threadsafe)
    """

    def __init__(
        self,
        config: BehaviorConfig,
        logger: logging.Logger,
        *,
        velocity_halt: Callable[[], Awaitable[bool]] | None = None,
        safe_hold: Callable[[], Awaitable[bool]] | None = None,
        milestone_callback: MilestoneCallback | None = None,
    ) -> None:
        self._config = config
        self._logger = logger
        self._slog: StructuredLogger = get_logger(config.behavior_type + "_coordinator")

        # Milestone emission
        self._milestone_callback = milestone_callback
        self._milestone_seq: int = 0
        self._milestones_dropped_total: int = 0
        self._milestones_last_drop_time: float = 0.0  # monotonic; 0.0 = no drops

        # Stop escalation callables (levels b and c) — mandatory (CP-016)
        async def _noop_halt() -> bool:
            return True
        self._velocity_halt = velocity_halt or _noop_halt
        self._safe_hold = safe_hold or _noop_halt
        self._last_stop_attempts: list[dict[str, Any]] = []

        # Event queue (serialized processing)
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=config.event_queue_size)

        # State machine (written by event loop only)
        self._state: BehaviorState = BehaviorState.IDLE
        self._session_id: str = ""
        self._turn_id: str = ""
        self._start_time: float = 0.0
        self._state_entry_time: float = 0.0
        self._pause_count: int = 0

        # Latched diagnostics (persist until new session reaches RUNNING)
        self._last_error: ErrorInfo | None = None
        self._last_session: LastSessionInfo | None = None

        # Thread safety for snapshot reads
        self._snapshot_lock = threading.Lock()

        # Readiness + component health cache (CP-003: wrapped in FreshValue for SI-5.3)
        self._readiness = ReadinessSnapshot(overall="UNAVAILABLE", timestamp=time.monotonic())
        self._component_health: dict[str, FreshValue[ComponentHealth]] = {}

        # Background tasks + timers
        self._event_loop_task: asyncio.Task | None = None
        self._health_monitor_task: asyncio.Task | None = None
        self._budget_tasks: dict[str, asyncio.Task] = {}

        # Health counters
        self._counters = HealthCounters(uptime_start=time.monotonic())

        # Lifecycle
        self._shutdown = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._http_session: aiohttp.ClientSession | None = None

        # Independent liveness watchdog (CP-002, SI-1.4, SI-4.1)
        self._watchdog = WatchdogThread(
            name=f"{config.behavior_type}_coordinator",
            interval_sec=2.0,
            on_timeout=self._on_watchdog_timeout,
            logger=self._slog,
        )

        # CP-005: reconciliation trackers registered by subclasses
        self._reconciliation_trackers: list[Any] = []

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start event loop and health monitor. Call from asyncio context."""
        self._shutdown = False
        self._loop = asyncio.get_running_loop()
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._config.health_http_timeout_sec)
        )
        await self._on_start()
        self._event_loop_task = asyncio.create_task(self._event_loop())
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self._watchdog.start()
        self._slog.info("coordinator_started", behavior_type=self._config.behavior_type)

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._shutdown = True
        self._watchdog.stop()

        # Cancel budget timers
        for task in self._budget_tasks.values():
            task.cancel()
        self._budget_tasks.clear()

        # Cancel main tasks
        for task in (self._event_loop_task, self._health_monitor_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._event_loop_task = None
        self._health_monitor_task = None

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

        await self._on_shutdown()
        self._slog.info("coordinator_shutdown", behavior_type=self._config.behavior_type)

    # =========================================================================
    # Public Interface (called by HTTP handlers)
    # =========================================================================

    async def request_start(
        self,
        session_id: str,
        turn_id: str,
        trace_ctx: Any | None = None,
        **params: Any,
    ) -> dict:
        """Handle behavior start request. Enqueues and awaits result.

        Clamps overridable budgets in params to their max ceilings.
        """
        # Clamp overridable budgets
        cfg = self._config
        for budget_key, max_key in (
            ("running_budget_sec", "running_budget_max_sec"),
            ("starting_budget_sec", "starting_budget_max_sec"),
            ("grace_budget_sec", "grace_budget_max_sec"),
        ):
            if budget_key in params:
                params[budget_key] = min(params[budget_key], getattr(cfg, max_key))

        loop = asyncio.get_running_loop()
        result_future = loop.create_future()

        event = _StartRequest(
            session_id=session_id,
            turn_id=turn_id,
            params=params,
            trace_ctx=trace_ctx,
            result_future=result_future,
        )

        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            self._counters.event_queue_drops += 1
            return {
                "success": False,
                "error": "Coordinator overloaded",
                "error_code": "UNAVAILABLE",
                "event_queue_depth": self._event_queue.qsize(),
            }

        return await result_future

    async def request_stop(
        self,
        session_id: str,
        turn_id: str,
        trace_ctx: Any | None = None,
    ) -> dict:
        """Handle behavior stop request. Blocks until CANCELLED.

        Hard ceiling: cancel_timeout_sec + 1.0s. If ceiling expires,
        returns STOPPING with a warning so caller can poll.
        """
        loop = asyncio.get_running_loop()
        result_future = loop.create_future()

        event = StopRequest(
            session_id=session_id,
            turn_id=turn_id,
            trace_ctx=trace_ctx,
            result_future=result_future,
        )

        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            self._counters.event_queue_drops += 1
            return {
                "success": False,
                "error": "Coordinator overloaded",
                "error_code": "UNAVAILABLE",
                "event_queue_depth": self._event_queue.qsize(),
            }

        # Block until CANCELLED with hard ceiling
        hard_ceiling = self._config.cancel_timeout_sec + 1.0
        try:
            return await asyncio.wait_for(result_future, timeout=hard_ceiling)
        except asyncio.TimeoutError:
            return {
                "success": True,
                "state": "STOPPING",
                "warning": "stop_in_progress",
            }

    def get_status(self) -> dict:
        """Get current behavior status (thread-safe snapshot read)."""
        with self._snapshot_lock:
            now = time.monotonic()
            state_name = self._config.state_aliases.get(self._state, self._state.value)

            result: dict[str, Any] = {
                "state": state_name,
                "session_id": self._session_id,
                "time_in_state_sec": round(now - self._state_entry_time, 1) if self._state_entry_time else 0.0,
                "readiness": self._readiness.overall,
                "pause_count": self._pause_count,
                "behavior_type": self._config.behavior_type,
                "extensions": {
                    self._config.behavior_type: self._get_status_extensions()
                },
            }

            if self._last_session:
                result["last_session"] = self._last_session.to_dict()

            if self._last_error:
                result["last_error"] = {
                    "code": self._last_error.code,
                    "detail": self._last_error.detail,
                    "timestamp": self._last_error.timestamp,
                }

            return result

    def get_readiness(self) -> ReadinessSnapshot:
        """Get current readiness snapshot (thread-safe deep copy)."""
        with self._snapshot_lock:
            return ReadinessSnapshot(
                overall=self._readiness.overall,
                components=dict(self._readiness.components),
                blocking_issues=list(self._readiness.blocking_issues),
                timestamp=self._readiness.timestamp,
            )

    def get_health(self) -> dict:
        """Get health counters for GET /<behavior>/health endpoint."""
        now = time.monotonic()
        c = self._counters
        with self._snapshot_lock:
            current_state = self._config.state_aliases.get(self._state, self._state.value)
            state_duration = round(now - self._state_entry_time, 1) if self._state_entry_time else 0.0

        return {
            "uptime_sec": round(now - c.uptime_start, 1),
            "sessions_started_total": c.sessions_started,
            "sessions_succeeded_total": c.sessions_succeeded,
            "sessions_failed_total": c.sessions_failed,
            "sessions_cancelled_total": c.sessions_cancelled,
            "pauses_total": c.pauses,
            "stop_requests_total": c.stop_requests,
            "stop_escalations_total": c.stop_escalations,
            "event_queue_drops_total": c.event_queue_drops,
            "exceptions_in_event_loop_total": c.exceptions_in_event_loop,
            "unknown_events_total": c.unknown_events,
            "health_http_failures_total": c.health_http_failures,
            "current_state": current_state,
            "current_state_duration_sec": state_duration,
            "event_queue_depth": self._event_queue.qsize(),
            "last_session_outcome": self._last_session.state if self._last_session else "",
            "last_stop_attempts": list(self._last_stop_attempts),
            "milestones_dropped_total": self._milestones_dropped_total,
            "milestones_last_drop_time": self._milestones_last_drop_time,
            "watchdog": self._watchdog.get_health(),
            # CP-005: reconciliation health for registered trackers
            "reconciliation": [
                t.as_health_dict() for t in self._reconciliation_trackers
            ],
        }

    # =========================================================================
    # Debug / Test Helpers
    # =========================================================================

    def debug_state(self) -> dict[str, Any]:
        """Inspection helper for tests. Returns raw (unaliased) state + session info."""
        with self._snapshot_lock:
            return {
                "state": self._state.value,
                "session_id": self._session_id,
                "start_time": self._start_time,
                "pause_count": self._pause_count,
                "last_error": self._last_error,
            }

    def debug_timers(self) -> dict[str, bool]:
        """Inspection helper for tests. Returns map of timer_name -> is_active."""
        return {
            name: not task.done()
            for name, task in self._budget_tasks.items()
        }

    def debug_config(self) -> dict[str, float]:
        """Inspection helper for tests. Returns budget/linger values from config."""
        c = self._config
        return {
            "starting_budget_sec": c.starting_budget_sec,
            "running_budget_default_sec": c.running_budget_default_sec,
            "grace_budget_sec": c.grace_budget_sec,
            "cancel_timeout_sec": c.cancel_timeout_sec,
            "failed_linger_sec": c.failed_linger_sec,
            "cancelled_linger_sec": c.cancelled_linger_sec,
            "succeeded_linger_sec": c.succeeded_linger_sec,
        }

    # =========================================================================
    # ROS Callback Interface (non-blocking enqueue)
    # =========================================================================

    def _enqueue_threadsafe(self, event: Any) -> None:
        """Enqueue event from a non-asyncio thread (e.g. ROS spin thread).

        asyncio.Queue is not thread-safe — uses call_soon_threadsafe to
        schedule put_nowait on the event loop thread.
        """
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._event_queue.put_nowait, event)
        except (RuntimeError, asyncio.QueueFull):
            self._counters.event_queue_drops += 1

    def _enqueue(self, event: Any) -> None:
        """Enqueue event from the asyncio thread."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            self._counters.event_queue_drops += 1

    # =========================================================================
    # Milestone Emission
    # =========================================================================

    def _emit_milestone(self, milestone: MilestoneBase) -> None:
        """Emit a milestone event to the registered callback.

        Thread-safe: always uses call_soon_threadsafe to schedule the
        callback on the event loop. Never raises to caller (MS-4).
        """
        if self._milestone_callback is None:
            return
        try:
            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._milestone_callback, milestone)
            else:
                # Loop not available — direct call (startup/test scenarios)
                self._milestone_callback(milestone)
        except RuntimeError:
            # Loop closed during shutdown
            pass
        except Exception:
            self._milestones_dropped_total += 1
            self._milestones_last_drop_time = time.monotonic()

    def _make_milestone(self, cls: type, **kwargs: Any) -> MilestoneBase:
        """Factory: create a milestone with common base fields populated."""
        self._milestone_seq += 1
        return cls(
            session_id=self._session_id,
            behavior_type=self._config.behavior_type,
            monotonic_ts=time.monotonic(),
            wall_time_ms=int(time.time() * 1000),
            sequence=self._milestone_seq,
            **kwargs,
        )

    # =========================================================================
    # Event Loop (serialized processing)
    # =========================================================================

    async def _event_loop(self) -> None:
        """Main event loop — processes all events sequentially."""
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=0.1
                )
            except asyncio.TimeoutError:
                self._watchdog.heartbeat()
                continue
            except asyncio.CancelledError:
                return

            try:
                await self._dispatch_event(event)
            except Exception as e:
                self._counters.exceptions_in_event_loop += 1
                self._logger.error(f"Event loop error processing {type(event).__name__}: {e}")
                self._slog.error("event_loop_exception",
                    event_type=type(event).__name__,
                    error=str(e)[:200],
                )
            self._watchdog.heartbeat()

    async def _dispatch_event(self, event: Any) -> None:
        """Route events to handlers. Priority: stop > start > health > budget > behavior."""
        # Priority 1: StopRequest (interrupt-class)
        if isinstance(event, StopRequest):
            await self._handle_base_stop(event)
            return
        # Priority 2: Base infrastructure events
        if isinstance(event, _StartRequest):
            await self._handle_base_start(event)
            return
        if isinstance(event, HealthEvent):
            self._handle_health_event(event)
            return
        if isinstance(event, BudgetExpired):
            if event.which in _BASE_BUDGET_NAMES:
                await self._handle_budget_expired(event)
            else:
                handled = await self._dispatch_behavior_event(event)
                if not handled:
                    self._logger.warning(f"Unknown budget timer expired: {event.which}")
                    self._counters.unknown_events += 1
            return
        # Priority 3: Behavior-specific events
        handled = await self._dispatch_behavior_event(event)
        if not handled:
            self._logger.warning(f"Unrecognized event type: {type(event).__name__}")
            self._counters.unknown_events += 1

    # =========================================================================
    # Base Event Handlers
    # =========================================================================

    async def _handle_base_start(self, event: _StartRequest) -> None:
        """Process start request with readiness and gate checks."""
        state = self._state

        # If already active or stopping, return current state (idempotent)
        if state in ACTIVE_STATES or state == BehaviorState.STOPPING:
            result = {
                "success": True,
                "state": self._config.state_aliases.get(state, state.value),
                "session_id": self._session_id,
            }
            if event.result_future and not event.result_future.done():
                event.result_future.set_result(result)
            return

        # If in a terminal state waiting for auto-clear, also return current state
        if state in TERMINAL_STATES:
            result = {
                "success": True,
                "state": self._config.state_aliases.get(state, state.value),
                "session_id": self._session_id,
            }
            if event.result_future and not event.result_future.done():
                event.result_future.set_result(result)
            return

        # Check readiness
        readiness = self._evaluate_readiness()
        with self._snapshot_lock:
            self._readiness = readiness

        if readiness.overall == "UNAVAILABLE":
            result = {
                "success": False,
                "error": f"System not ready: {', '.join(readiness.blocking_issues)}",
                "error_code": f"{self._config.behavior_type.upper()}_NOT_READY",
                "state": state.value,
                "readiness": readiness.overall,
                "blocking_issues": readiness.blocking_issues,
            }
            if event.result_future and not event.result_future.done():
                event.result_future.set_result(result)
            return

        # Pre-start gate
        gate_result = self._check_pre_start_gate()
        if not gate_result.get("pass", False):
            result = {
                "success": False,
                "error": gate_result.get("reason", "Pre-start gate failed"),
                "error_code": gate_result.get("error_code", "GATE_FAILED"),
                "instruction": gate_result.get("instruction", ""),
                "state": state.value,
            }
            if event.result_future and not event.result_future.done():
                event.result_future.set_result(result)
            return

        # Reset base session fields (do NOT clear _last_error/_last_session)
        self._session_id = event.session_id
        self._turn_id = event.turn_id
        self._start_time = time.monotonic()
        self._pause_count = 0
        self._milestone_seq = 0  # reset per-session counter
        self._counters.sessions_started += 1

        # Emit CommandAccepted after gates pass
        self._emit_milestone(self._make_milestone(CommandAccepted))

        # Delegate to subclass
        result = await self._on_start_approved(event)
        if event.result_future and not event.result_future.done():
            event.result_future.set_result(result)

    async def _handle_base_stop(self, event: StopRequest) -> None:
        """Process stop request — interrupt-class, always works.

        Blocks until CANCELLED, then resolves the Future.
        Budget: _STOP_ESCALATION_BUDGET_SEC (default ceiling; actual ceiling
        derived dynamically in _stop_escalation() from BehaviorConfig).
        """
        # Budget: _STOP_ESCALATION_BUDGET_SEC
        state = self._state

        # If IDLE or terminal, return current state immediately
        if state == BehaviorState.IDLE or state in TERMINAL_STATES:
            result = {
                "success": True,
                "state": self._config.state_aliases.get(state, state.value),
                "previous_state": self._config.state_aliases.get(state, state.value),
            }
            if event.result_future and not event.result_future.done():
                event.result_future.set_result(result)
            return

        previous_state = state
        self._counters.stop_requests += 1

        # Transition to STOPPING
        self._transition_to(BehaviorState.STOPPING, "explicit_stop")

        # Cancel all budget timers
        self._cancel_all_budget_timers()

        # Run stop escalation (returns False if ceiling exceeded)
        completed = await self._stop_escalation()

        if not completed:
            error_code = f"{self._config.behavior_type.upper()}_STOP_TIMEOUT"
            self._transition_to_failed(error_code, "stop escalation ceiling exceeded")
            result = {
                "success": True,
                "state": "FAILED",
                "previous_state": self._config.state_aliases.get(previous_state, previous_state.value),
                "error_code": error_code,
            }
            if event.result_future and not event.result_future.done():
                event.result_future.set_result(result)
            return

        # Transition to CANCELLED
        self._transition_to(BehaviorState.CANCELLED, "explicit_stop")

        # MS-3: exactly one TerminalOutcome per terminal state
        self._emit_milestone(self._make_milestone(TerminalOutcome,
            outcome="CANCELLED",
            reason_code="explicit_stop",
            error_detail="",
        ))

        # Latch session info
        self._last_session = LastSessionInfo(
            state="CANCELLED",
            error_code="",
            error_detail="",
            ended_at=time.monotonic(),
            reason="explicit_stop",
        )

        # Start auto-clear timer
        self._start_budget_timer("cancelled_linger", self._config.cancelled_linger_sec)
        self._counters.sessions_cancelled += 1

        self._slog.info("state_transition",
            entity_type="behavior",
            entity_id=self._session_id or self._config.behavior_type,
            old_state=self._config.state_aliases.get(previous_state, previous_state.value),
            new_state="CANCELLED",
            session_id=self._session_id,
            reason="explicit_stop",
        )

        result = {
            "success": True,
            "state": "CANCELLED",
            "previous_state": self._config.state_aliases.get(previous_state, previous_state.value),
        }
        if event.result_future and not event.result_future.done():
            event.result_future.set_result(result)

    def _handle_health_event(self, event: HealthEvent) -> None:
        """Update component health cache and notify subclass."""
        # CP-003 (SI-5.3): wrap health in FreshValue at acquisition site.
        # max_age_sec = health_staleness_sec from config.
        health = ComponentHealth(
            status=event.status,
            detail=event.detail,
            last_seen=event.timestamp,
        )
        self._component_health[event.component] = FreshValue(
            value=health,
            timestamp=event.timestamp,
            max_age_sec=self._config.health_staleness_sec,
        )
        self._on_health_event(event)

    async def _handle_budget_expired(self, event: BudgetExpired) -> None:
        """Handle budget timer expiry for base-managed timers."""
        which = event.which
        state = self._state

        if which == "starting" and state == BehaviorState.STARTING:
            # STOPPING → escalation → FAILED
            self._transition_to(BehaviorState.STOPPING, "starting_budget_expired")
            self._cancel_all_budget_timers()
            self._counters.stop_requests += 1
            completed = await self._stop_escalation()
            if not completed:
                error_code = f"{self._config.behavior_type.upper()}_STOP_TIMEOUT"
            else:
                error_code = f"{self._config.behavior_type.upper()}_START_TIMEOUT"
            self._transition_to_failed(error_code, "starting budget expired")

        elif which == "running" and state == BehaviorState.RUNNING:
            # STOPPING → escalation → CANCELLED (time budget = operational limit, not failure)
            self._transition_to(BehaviorState.STOPPING, "running_budget_expired")
            self._cancel_all_budget_timers()
            self._counters.stop_requests += 1
            completed = await self._stop_escalation()
            if not completed:
                error_code = f"{self._config.behavior_type.upper()}_STOP_TIMEOUT"
                self._transition_to_failed(error_code, "stop escalation ceiling exceeded during running budget expiry")
                return
            self._transition_to(BehaviorState.CANCELLED, "running_budget_expired")

            # MS-3: exactly one TerminalOutcome per terminal state
            self._emit_milestone(self._make_milestone(TerminalOutcome,
                outcome="CANCELLED",
                reason_code="TIME_BUDGET_EXPIRED",
                error_detail="session running budget expired",
            ))

            self._last_session = LastSessionInfo(
                state="CANCELLED",
                error_code="TIME_BUDGET_EXPIRED",
                error_detail="session running budget expired",
                ended_at=time.monotonic(),
                reason="running_budget_expired",
            )
            self._start_budget_timer("cancelled_linger", self._config.cancelled_linger_sec)
            self._counters.sessions_cancelled += 1

        elif which == "grace" and state == BehaviorState.PAUSED:
            # STOPPING → escalation → FAILED
            self._transition_to(BehaviorState.STOPPING, "grace_budget_expired")
            self._cancel_all_budget_timers()
            self._counters.stop_requests += 1
            completed = await self._stop_escalation()
            if not completed:
                error_code = f"{self._config.behavior_type.upper()}_STOP_TIMEOUT"
            else:
                error_code = f"{self._config.behavior_type.upper()}_GRACE_EXPIRED"
            self._transition_to_failed(error_code, "grace budget expired")

        elif which == "failed_linger" and state == BehaviorState.FAILED:
            self._transition_to(BehaviorState.IDLE, "auto_clear")

        elif which == "cancelled_linger" and state == BehaviorState.CANCELLED:
            self._transition_to(BehaviorState.IDLE, "auto_clear")

        elif which == "succeeded_linger" and state == BehaviorState.SUCCEEDED:
            self._transition_to(BehaviorState.IDLE, "auto_clear")

    # =========================================================================
    # State Transition Helpers
    # =========================================================================

    def _on_watchdog_timeout(self, name: str, elapsed_sec: float) -> None:
        """Called from the watchdog thread when the event loop stalls.

        Marshals to the asyncio event loop via call_soon_threadsafe (SI-4.1).
        If the loop is unavailable (already dead), transitions directly.
        """
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(
                    self._force_failed_from_watchdog, elapsed_sec
                )
            except RuntimeError:
                # Loop closed during shutdown
                pass
        else:
            self._force_failed_from_watchdog(elapsed_sec)

    def _force_failed_from_watchdog(self, elapsed_sec: float) -> None:
        """Transition to FAILED due to watchdog timeout.

        Called on the event loop thread (via call_soon_threadsafe) or
        directly if the loop is unavailable.
        """
        if self._state in TERMINAL_STATES or self._state == BehaviorState.IDLE:
            return
        error_code = f"{self._config.behavior_type.upper()}_WATCHDOG_TIMEOUT"
        detail = f"Event loop stalled for {elapsed_sec:.1f}s"
        self._transition_to_failed(error_code, detail)

    def _transition_to(self, new_state: BehaviorState, reason: str) -> None:
        """Generic state transition with snapshot update and logging."""
        with self._snapshot_lock:
            old_state = self._state
            self._state = new_state
            self._state_entry_time = time.monotonic()

            # Clear latched diagnostics when new session confirmed active
            if new_state == BehaviorState.RUNNING:
                self._last_error = None
                self._last_session = None

        self._slog.info("state_transition",
            entity_type="behavior",
            entity_id=self._session_id or self._config.behavior_type,
            old_state=self._config.state_aliases.get(old_state, old_state.value),
            new_state=self._config.state_aliases.get(new_state, new_state.value),
            session_id=self._session_id,
            reason=reason,
        )

        # MS-1: PhaseChanged emitted after state write is committed
        self._emit_milestone(self._make_milestone(PhaseChanged,
            from_state=old_state.value,
            to_state=new_state.value,
            alias=self._config.state_aliases.get(new_state, new_state.value),
            reason=reason,
        ))

    def _transition_to_failed(self, error_code: str, detail: str) -> None:
        """Transition to FAILED with latched diagnostics."""
        now = time.monotonic()

        self._last_error = ErrorInfo(
            code=error_code,
            detail=detail,
            timestamp=now,
        )
        self._last_session = LastSessionInfo(
            state="FAILED",
            error_code=error_code,
            error_detail=detail,
            ended_at=now,
            reason=detail,
        )

        # Emit Fault milestone before transition
        self._emit_milestone(self._make_milestone(Fault,
            error_code=error_code,
            severity="CRITICAL",
            recoverable=False,
        ))

        # Cancel all active budget timers
        self._cancel_all_budget_timers()

        self._transition_to(BehaviorState.FAILED, error_code)

        # MS-3: exactly one TerminalOutcome per terminal state
        self._emit_milestone(self._make_milestone(TerminalOutcome,
            outcome="FAILED",
            reason_code=error_code,
            error_detail=detail,
        ))

        # Start failed linger timer
        self._start_budget_timer("failed_linger", self._config.failed_linger_sec)
        self._counters.sessions_failed += 1

    def _transition_to_succeeded(self, reason: str) -> None:
        """Transition to SUCCEEDED with latched session info."""
        self._last_session = LastSessionInfo(
            state="SUCCEEDED",
            error_code="",
            error_detail="",
            ended_at=time.monotonic(),
            reason=reason,
        )

        self._cancel_all_budget_timers()
        self._transition_to(BehaviorState.SUCCEEDED, reason)

        # MS-3: exactly one TerminalOutcome per terminal state
        self._emit_milestone(self._make_milestone(TerminalOutcome,
            outcome="SUCCEEDED",
            reason_code=reason,
            error_detail="",
        ))

        self._start_budget_timer("succeeded_linger", self._config.succeeded_linger_sec)
        self._counters.sessions_succeeded += 1

    # =========================================================================
    # Budget Timers
    # =========================================================================

    def _start_budget_timer(self, name: str, duration_sec: float) -> None:
        """Start a named budget timer. Cancels existing timer with same name."""
        self._cancel_budget_timer(name)

        async def _timer() -> None:
            await asyncio.sleep(duration_sec)
            self._enqueue(BudgetExpired(which=name))

        self._budget_tasks[name] = asyncio.create_task(_timer())

    def _cancel_budget_timer(self, name: str) -> None:
        """Cancel a named budget timer if running."""
        task = self._budget_tasks.pop(name, None)
        if task and not task.done():
            task.cancel()

    def _cancel_all_budget_timers(self) -> None:
        """Cancel all active budget timers."""
        for task in self._budget_tasks.values():
            if not task.done():
                task.cancel()
        self._budget_tasks.clear()

    # =========================================================================
    # Stop Escalation
    # =========================================================================

    async def _stop_escalation(self) -> bool:
        """Stop escalation ladder with observability and hard ceiling.

        Level (a): Behavior-specific cancel (_cancel_behavior).
        Level (b): Velocity halt (optional, via velocity_halt callable).
        Level (c): SAFE_HOLD (optional, via safe_hold callable).

        Records ordered _last_stop_attempts for test assertions and monitoring.
        Hard ceiling is derived from sum(per-level timeouts) + epsilon.

        Returns True if completed within ceiling, False if ceiling exceeded.
        """
        self._last_stop_attempts = []

        # Build level list — all levels mandatory (CP-016)
        levels: list[tuple[str, Callable[[], Awaitable[bool]], float]] = [
            ("a", self._cancel_behavior, self._config.cancel_timeout_sec),
            ("b", self._velocity_halt, self._config.velocity_halt_timeout_sec),
            ("c", self._safe_hold, self._config.safe_hold_timeout_sec),
        ]

        # Derive hard ceiling from per-level timeouts + margin (CP-008, SI-3.3)
        # With defaults: 0.5 + 0.3 + 0.3 + 0.9 = 2.0s total ceiling
        ceiling = sum(t for _, _, t in levels) + _STOP_CEILING_MARGIN_SEC

        try:
            await asyncio.wait_for(
                self._run_escalation_levels(levels), timeout=ceiling
            )
            return True
        except asyncio.TimeoutError:
            self._last_stop_attempts.append({
                "level": "ceiling", "result": "exceeded",
                "elapsed_ms": int(ceiling * 1000),
            })
            self._counters.stop_escalations += 1
            self._slog.error("stop_ceiling_exceeded",
                ceiling_sec=ceiling,
                attempts=len(self._last_stop_attempts),
                session_id=self._session_id,
            )
            return False

    async def _run_escalation_levels(
        self,
        levels: list[tuple[str, Callable[[], Awaitable[bool]], float]],
    ) -> None:
        """Execute stop escalation levels sequentially with per-level timeouts."""
        for level_name, fn, timeout in levels:
            t0 = time.monotonic()
            try:
                result = await asyncio.wait_for(fn(), timeout=timeout)
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                result_str = "ok" if result else "rejected"
            except asyncio.TimeoutError:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                result_str = "timeout"
                self._counters.stop_escalations += 1
            except Exception as e:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                result_str = "error"
                self._counters.stop_escalations += 1
                self._slog.error("stop_level_error",
                    escalation_level=level_name,
                    error=str(e)[:200],
                    session_id=self._session_id,
                )

            attempt = {
                "level": level_name,
                "result": result_str,
                "elapsed_ms": elapsed_ms,
            }
            self._last_stop_attempts.append(attempt)
            self._slog.info("stop_level_attempted",
                escalation_level=level_name,
                result=result_str,
                elapsed_ms=elapsed_ms,
                session_id=self._session_id,
            )

            self._emit_milestone(self._make_milestone(StopEscalationLevel,
                level_name=level_name,
                outcome=result_str,
                elapsed_ms=elapsed_ms,
            ))

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    async def _health_monitor_loop(self) -> None:
        """Background health monitor. Polls configured endpoints and updates readiness."""
        while not self._shutdown:
            try:
                for component, url in self._config.health_endpoints.items():
                    await self._check_gateway_health(component, url)

                readiness = self._evaluate_readiness()
                with self._snapshot_lock:
                    self._readiness = readiness

            except asyncio.CancelledError:
                return
            except Exception as e:
                self._logger.warning(f"Health monitor error: {e}")

            await asyncio.sleep(self._config.health_poll_interval_sec)

    async def _check_gateway_health(self, component: str, url: str) -> None:
        """Check a gateway's health via HTTP GET."""
        now = time.monotonic()

        if not self._http_session or self._http_session.closed:
            self._counters.health_http_failures += 1
            self._enqueue(HealthEvent(
                component=component,
                status="MISSING",
                detail="HTTP session not available",
                timestamp=now,
            ))
            return

        try:
            async with self._http_session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    status, detail = self._classify_health_response(component, resp.status, data)
                else:
                    status = "ERROR"
                    detail = f"HTTP {resp.status}"

                self._enqueue(HealthEvent(
                    component=component,
                    status=status,
                    detail=detail,
                    timestamp=now,
                ))

        except asyncio.TimeoutError:
            self._counters.health_http_failures += 1
            self._enqueue(HealthEvent(
                component=component, status="ERROR", detail="timeout", timestamp=now,
            ))
        except aiohttp.ClientConnectorError:
            self._counters.health_http_failures += 1
            self._enqueue(HealthEvent(
                component=component, status="ERROR", detail="connection refused", timestamp=now,
            ))
        except Exception as e:
            self._counters.health_http_failures += 1
            self._enqueue(HealthEvent(
                component=component, status="ERROR", detail=str(e)[:100], timestamp=now,
            ))

    def _evaluate_readiness(self) -> ReadinessSnapshot:
        """Evaluate system readiness from cached component health.

        CP-003 (SI-5.3): uses FreshValue.is_fresh for staleness detection
        instead of manual timestamp arithmetic. If component health data
        exceeds the declared freshness bound, it is treated as MISSING.
        """
        now = time.monotonic()
        components: dict[str, ComponentHealth] = {}
        blocking: list[str] = []
        degraded: list[str] = []

        for name, fv_health in self._component_health.items():
            if not fv_health.is_fresh:
                age = fv_health.age_sec
                effective = ComponentHealth(
                    status="MISSING",
                    detail=f"stale ({age:.1f}s, limit {fv_health.max_age_sec}s)",
                    last_seen=fv_health.value.last_seen,
                )
            else:
                effective = fv_health.value

            components[name] = effective

            if effective.status in ("ERROR", "MISSING"):
                is_blocking = name in self._config.health_endpoint_blocking
                target = blocking if is_blocking else degraded
                target.append(f"{name}: {effective.detail}")
            elif effective.status == "DEGRADED":
                degraded.append(f"{name}: {effective.detail}")

        # Let subclass add behavior-specific component checks
        self._evaluate_behavior_readiness(components, blocking, degraded)

        overall = "UNAVAILABLE" if blocking else "DEGRADED" if degraded else "READY"
        return ReadinessSnapshot(
            overall=overall,
            components=components,
            blocking_issues=blocking,
            timestamp=now,
        )

    # =========================================================================
    # Abstract Methods (subclass MUST implement)
    # =========================================================================

    @abc.abstractmethod
    async def _on_start_approved(self, event: _StartRequest) -> dict:
        """Handle approved start request.

        Base has already validated readiness and gate, reset session fields,
        and incremented sessions_started. Subclass should:
        1. Transition to STARTING
        2. Initialize behavior-specific session state
        3. Start the starting budget timer
        4. Launch async acquisition if needed
        5. Return immediate response dict
        """
        ...

    @abc.abstractmethod
    async def _cancel_behavior(self) -> bool:
        """Level (a) stop — behavior-specific cancel.

        Called with timeout by _stop_escalation(). Return True on success.
        """
        ...

    @abc.abstractmethod
    def _check_pre_start_gate(self) -> dict:
        """Pre-start validation gate.

        Returns {"pass": True} or {"pass": False, "reason": ...,
        "error_code": ..., "instruction": ...}.
        """
        ...

    @abc.abstractmethod
    def _evaluate_behavior_readiness(
        self,
        components: dict[str, ComponentHealth],
        blocking: list[str],
        degraded: list[str],
    ) -> None:
        """Add behavior-specific component checks to readiness.

        Mutate blocking/degraded lists in-place. Called during readiness
        evaluation after base gateway checks.
        """
        ...

    @abc.abstractmethod
    def _get_status_extensions(self) -> dict[str, Any]:
        """Return flat dict of behavior-specific status fields.

        Will be nested under extensions.{behavior_type} in get_status().
        """
        ...

    @abc.abstractmethod
    async def _dispatch_behavior_event(self, event: Any) -> bool:
        """Handle behavior-specific events.

        Return True if handled, False if unknown. Also receives unknown
        BudgetExpired names (for custom behavior timers).
        """
        ...

    # =========================================================================
    # Optional Override Methods (default no-op)
    # =========================================================================

    async def _on_start(self) -> None:
        """Called during start() after aiohttp session creation, before event loop launch."""
        pass

    async def _on_shutdown(self) -> None:
        """Called during shutdown() after tasks cancelled and HTTP session closed."""
        pass

    def _classify_health_response(
        self, component: str, status_code: int, data: dict
    ) -> tuple[str, str]:
        """Classify an HTTP health response into (status, detail).

        Override for component-specific health parsing (e.g. motion_gateway
        mode/estop checks).
        """
        if status_code == 200:
            return ("OK", "healthy")
        return ("ERROR", f"HTTP {status_code}")

    def _on_health_event(self, event: HealthEvent) -> None:
        """Called when a health event is received. Override to trigger hard faults."""
        pass
