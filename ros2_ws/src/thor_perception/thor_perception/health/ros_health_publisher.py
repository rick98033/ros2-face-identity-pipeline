"""ROS health publisher mixin for Category C nodes (CP-004).

Provides counter tracking and periodic HealthStatus publication
at 1 Hz to /health/<node_name>. ros-gateway subscribes and aggregates.

Usage:
    class MyNode(Node, RosHealthPublisher):
        def __init__(self):
            super().__init__('my_node')
            self._health_init('my-component')
            ...

        def _health_ok(self) -> bool:
            return self._some_condition

        def destroy_node(self):
            self._health_destroy()
            super().destroy_node()
"""

import json
import time
from datetime import datetime, timezone
from typing import Optional


class RosHealthPublisher:
    """Mixin for ROS 2 nodes to publish CP-004 health status."""

    def _health_init(self, component: str) -> None:
        """Initialize health publishing. Call from __init__ after super().__init__."""
        self._health_component = component
        self._health_counters: dict[str, int] = {}
        self._health_last_error: Optional[str] = None
        self._health_last_error_ts: Optional[str] = None
        self._health_start_mono = time.monotonic()

        # Lazy import to avoid circular dependency at module level
        from thor_msgs.msg import HealthStatus
        self._HealthStatus = HealthStatus

        self._health_pub = self.create_publisher(
            HealthStatus,
            f"/health/{component}",
            10,
        )
        self._health_timer = self.create_timer(1.0, self._health_publish)

    def _health_increment(self, name: str, n: int = 1) -> None:
        """Increment a named counter."""
        self._health_counters[name] = self._health_counters.get(name, 0) + n

    def _health_record_error(self, code: str) -> None:
        """Record an error and increment errors_total."""
        self._health_last_error = code
        self._health_last_error_ts = datetime.now(timezone.utc).isoformat()
        self._health_increment("errors_total")

    def _health_ok(self) -> bool:
        """Override in subclass to provide health assessment. Default: True."""
        return True

    def _health_status(self) -> str:
        """Override in subclass for degraded state. Default: derives from _health_ok."""
        return "healthy" if self._health_ok() else "unhealthy"

    def _health_publish(self) -> None:
        """Publish HealthStatus message. Called by 1 Hz timer."""
        msg = self._HealthStatus()
        msg.component = self._health_component
        msg.uptime_ms = int((time.monotonic() - self._health_start_mono) * 1000)
        msg.ok = self._health_ok()
        msg.status = self._health_status()
        msg.counters_json = json.dumps(self._health_counters)
        msg.last_error = self._health_last_error or ""
        msg.last_error_ts = self._health_last_error_ts or ""
        msg.ts = datetime.now(timezone.utc).isoformat()
        self._health_pub.publish(msg)

    def _health_destroy(self) -> None:
        """Clean up health timer. Call from destroy_node() before super()."""
        if hasattr(self, '_health_timer') and self._health_timer is not None:
            self._health_timer.cancel()
