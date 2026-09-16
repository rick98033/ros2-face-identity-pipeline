"""Independent liveness watchdog (CP-002).

A reusable daemon thread that detects when monitored code has stalled
by tracking heartbeat() calls. If heartbeats stop, the watchdog fires
an on_timeout callback from an independent thread.

Design requirements:
  - SI-1.4:  Continuous self-check at declared intervals
  - SI-4.1: Safety layer independence (separate thread, no shared event loop)
  - SI-4.2: Testable in isolation (pure Python, no ROS, no asyncio)

Usage:
    from thor_behavior import WatchdogThread, WatchdogState

    def on_stall(name: str, elapsed_sec: float):
        log.warning("stalled", component=name, elapsed_sec=elapsed_sec)

    wd = WatchdogThread("my_loop", interval_sec=2.0, on_timeout=on_stall)
    wd.start()

    # In monitored code's main loop:
    while running:
        wd.heartbeat()
        do_work()

    wd.stop()
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from thor_telemetry import StructuredLogger


class WatchdogState(Enum):
    """Watchdog lifecycle states (SI-1.1: enumerated state set)."""
    UNKNOWN = "unknown"
    IDLE = "idle"
    MONITORING = "monitoring"
    TIMED_OUT = "timed_out"
    STOPPED = "stopped"


# CP-003: Lifecycle state sets for WatchdogState.
TERMINAL_STATES = frozenset({WatchdogState.STOPPED, WatchdogState.UNKNOWN})
ACTIVE_STATES = frozenset({WatchdogState.MONITORING})


class WatchdogThread:
    """Independent liveness watchdog running in a daemon thread.

    The monitored code calls heartbeat() at least every interval_sec.
    If heartbeats stop, on_timeout(name, elapsed_sec) fires from the
    watchdog thread. The watchdog is independent of the monitored code's
    event loop (SI-4.1).

    Args:
        name: Identifier for logging and health reporting.
        interval_sec: Maximum allowed gap between heartbeats. Must be > 0.
        on_timeout: Called when heartbeat gap exceeds interval_sec.
            Receives (name, elapsed_sec). Called from watchdog thread.
            Exceptions are caught and logged — watchdog survives.
        logger: Optional StructuredLogger for events. If None, events
            are silently skipped (tests can omit).
    """

    def __init__(
        self,
        name: str,
        interval_sec: float,
        on_timeout: Callable[[str, float], None],
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        if interval_sec <= 0:
            raise ValueError(f"interval_sec must be > 0, got {interval_sec}")

        self._name = name
        self._interval_ns = int(interval_sec * 1_000_000_000)
        self._on_timeout = on_timeout
        self._log = logger

        # Tick: check 4x per interval, capped at 250ms
        self._tick_sec = min(interval_sec / 4, 0.25)

        # State under lock
        self._lock = threading.Lock()
        self._state = WatchdogState.IDLE
        self._last_heartbeat_ns: Optional[int] = None
        self._timeout_count = 0
        self._stop_clean: Optional[bool] = None

        # Thread control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the watchdog thread. Transitions IDLE -> MONITORING."""
        with self._lock:
            if self._state != WatchdogState.IDLE:
                return
            self._last_heartbeat_ns = time.monotonic_ns()
            self._state = WatchdogState.MONITORING
            self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._run, name=f"watchdog-{self._name}", daemon=True
        )
        self._thread.start()
        if self._log:
            self._log.info("watchdog_started", component=self._name)

    def stop(self) -> bool:
        """Stop the watchdog thread.

        Returns True if the thread joined cleanly, False if join timed
        out (thread still running). Idempotent on STOPPED or IDLE.
        """
        with self._lock:
            if self._state in (WatchdogState.STOPPED, WatchdogState.IDLE):
                return True
            self._state = WatchdogState.STOPPED

        self._stop_event.set()
        if self._thread is not None:
            join_timeout = (self._interval_ns / 1_000_000_000) * 2
            self._thread.join(timeout=join_timeout)
            clean = not self._thread.is_alive()
        else:
            clean = True

        with self._lock:
            self._stop_clean = clean

        if self._log:
            self._log.info(
                "watchdog_stopped",
                component=self._name,
                stop_clean=clean,
            )
        if not clean and self._log:
            self._log.warning(
                "watchdog_join_timeout",
                component=self._name,
                msg="Thread did not join within timeout",
            )
        return clean

    def heartbeat(self) -> None:
        """Record a heartbeat from the monitored code.

        Thread-safe. Called from the monitored code's thread.
        """
        with self._lock:
            self._last_heartbeat_ns = time.monotonic_ns()
            if self._state == WatchdogState.TIMED_OUT:
                self._state = WatchdogState.MONITORING
                if self._log:
                    self._log.info(
                        "watchdog_recovered", component=self._name
                    )

    def get_health(self) -> dict:
        """Return health counters compatible with health endpoint convention."""
        with self._lock:
            now_ns = time.monotonic_ns()
            if self._last_heartbeat_ns is not None:
                age_sec = (now_ns - self._last_heartbeat_ns) / 1_000_000_000
            else:
                age_sec = None
            return {
                "name": self._name,
                "state": self._state.value,
                "timeout_count": self._timeout_count,
                "last_heartbeat_age_sec": age_sec,
                "is_alive": self._thread.is_alive() if self._thread else False,
                "stop_clean": self._stop_clean,
                "interval_sec": self._interval_ns / 1_000_000_000,
            }

    @property
    def state(self) -> WatchdogState:
        with self._lock:
            return self._state

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive() if self._thread else False

    @property
    def timeout_count(self) -> int:
        with self._lock:
            return self._timeout_count

    # ── Internal ──

    def _run(self) -> None:
        """Main watchdog loop. Runs in the daemon thread."""
        while not self._stop_event.wait(timeout=self._tick_sec):
            self._check_heartbeat()

    def _check_heartbeat(self) -> None:
        """Check if heartbeat has exceeded interval. Fire on_timeout if so."""
        with self._lock:
            if self._state != WatchdogState.MONITORING:
                return
            if self._last_heartbeat_ns is None:
                return
            now_ns = time.monotonic_ns()
            age_ns = now_ns - self._last_heartbeat_ns
            if age_ns <= self._interval_ns:
                return
            # Timeout detected
            self._state = WatchdogState.TIMED_OUT
            self._timeout_count += 1
            elapsed_sec = age_ns / 1_000_000_000

        # Call outside lock to avoid deadlock
        if self._log:
            self._log.warning(
                "watchdog_timeout",
                component=self._name,
                operation="heartbeat_check",
                error_code="WATCHDOG_TIMEOUT",
                elapsed_sec=round(elapsed_sec, 3),
                timeout_count=self._timeout_count,
            )
        try:
            self._on_timeout(self._name, elapsed_sec)
        except Exception:
            if self._log:
                self._log.error(
                    "watchdog_callback_error",
                    component=self._name,
                    msg="on_timeout callback raised",
                    exc_info=True,
                )
