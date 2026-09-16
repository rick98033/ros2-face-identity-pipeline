"""Shedding policy infrastructure (CP-032).

Declares priority tiers and shedding behavior for message queues under
backpressure. Currently structural — no consumer consults these policies.
Wire when a backpressure monitor is built.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class MessageTier(IntEnum):
    """Priority tier for queued messages (CP-032).

    Lower value = higher priority = shed last.
    """
    UNKNOWN = -1        # Uninitialized / unrecognized (CP-010)
    SAFETY_CRITICAL = 0  # Never shed (stop commands, safety alerts)
    URGENT = 1           # Time-sensitive operational (state transitions, commands)
    OPERATIONAL = 2      # Standard operational messages
    INFORMATIONAL = 3    # Status updates, telemetry
    DEBUG = 4            # Diagnostics, verbose logging


class OverflowBehavior(IntEnum):
    """What to do when queue capacity is exceeded (CP-032)."""
    UNKNOWN = -1    # Uninitialized / unrecognized (CP-010)
    DROP_LOWEST = 0  # Shed lowest-priority messages first
    DROP_OLDEST = 1  # Shed oldest messages regardless of tier
    BLOCK = 2        # Block sender until space available


@dataclass(frozen=True)
class SheddingPolicy:
    """Shedding policy for a message queue (CP-032).

    Declares how a component's queue should behave under backpressure.
    """
    component: str
    queue_description: str
    capacity: int
    tier_thresholds: dict[MessageTier, float | None]
    overflow_behavior: str
    shed_counter_name: str


# Default shedding policy for perception pipeline event queue (CP-032)
SHEDDING_POLICY = SheddingPolicy(
    component="perception_pipeline",
    queue_description="Perception event queue (detections, tracks, state transitions)",
    capacity=256,
    tier_thresholds={
        MessageTier.SAFETY_CRITICAL: None,  # Never shed
        MessageTier.URGENT: 0.9,            # Shed above 90% capacity
        MessageTier.OPERATIONAL: 0.75,      # Shed above 75% capacity
        MessageTier.INFORMATIONAL: 0.5,     # Shed above 50% capacity
        MessageTier.DEBUG: 0.25,            # Shed above 25% capacity
    },
    overflow_behavior="drop_lowest",
    shed_counter_name="perception_messages_shed_total",
)

# Message type to tier mapping (CP-032)
MESSAGE_TIER_MAP: dict[str, MessageTier] = {
    "stop_command": MessageTier.SAFETY_CRITICAL,
    "safety_alert": MessageTier.SAFETY_CRITICAL,
    "state_transition": MessageTier.URGENT,
    "auth_change": MessageTier.URGENT,
    "person_detection": MessageTier.OPERATIONAL,
    "face_match": MessageTier.OPERATIONAL,
    "track_update": MessageTier.OPERATIONAL,
    "health_status": MessageTier.INFORMATIONAL,
    "timing_report": MessageTier.INFORMATIONAL,
    "debug_overlay": MessageTier.DEBUG,
}
