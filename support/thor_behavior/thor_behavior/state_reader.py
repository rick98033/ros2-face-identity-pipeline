"""StateReader protocol for truthfulness guards.

Defines the read-only interface that the VoiceOrchestrator uses to check
coordinator state before allowing speech. Extensible: future behaviors
add properties (e.g., localization_ok) without changing the engine —
the engine checks all declared keys from truthfulness_requires.

Thread-safe: implementations acquire the coordinator's snapshot lock.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StateReader(Protocol):
    """Read-only view of coordinator state for truthfulness checks."""

    @property
    def state(self) -> str:
        """Current behavior state (e.g., 'RUNNING', 'PAUSED')."""
        ...

    @property
    def session_id(self) -> str:
        """Active session ID, or '' if idle."""
        ...

    @property
    def motion_lease_active(self) -> bool:
        """True if the coordinator currently holds motion authority."""
        ...

    @property
    def error_code(self) -> str:
        """Last error code, or '' if no error."""
        ...
