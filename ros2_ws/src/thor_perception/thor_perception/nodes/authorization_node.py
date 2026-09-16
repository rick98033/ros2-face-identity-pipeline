#!/usr/bin/env python3
"""Authorization Manager Node - Phase 3.3

Binds a user to a persistent person track_id using face evidence.
Publishes /world/authorized_target as the single source of truth for Phase 4 follow-me.

State Machine:
    UNAUTHORIZED -> ACQUIRING -> AUTHORIZED -> SUSPENDED

Key contracts:
    - Never silently switch authorized_track_id while AUTHORIZED
    - 2-of-N acquisition rule: 2 consistent face matches within window
    - Confidence decays without face evidence
    - Ambiguity triggers SUSPENDED, not silent switch

Usage:
    ros2 run thor_perception authorization_node
    ros2 launch thor_perception identity_pipeline.launch.py
"""

import sys
import time
import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from threading import Lock
from typing import Optional, Dict, List, Tuple

from thor_behavior import (
    WatchdogThread,
    DegradationPolicy, DegradedState, DepPolicy, FailureMode, RecoveryTarget,
    ResponseType,
)

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from std_msgs.msg import Header

from thor_msgs.msg import (
    AuthorizedTarget,
    PersonTracks,
    PersonTrack,
    FaceTracks,
    FaceTrack,
    FaceIdentityCandidates,
    FaceIdentityCandidate,
)
from thor_msgs.srv import RequestFollow, CancelFollow
from thor_telemetry import get_logger as _get_structured_logger, ErrorCode
from thor_telemetry import BoundaryContract, BoundaryType, ErrorBehavior
from thor_telemetry import RetentionDeclaration
from thor_perception.health.ros_health_publisher import RosHealthPublisher


_structured_logger = _get_structured_logger("authorization_node")

# --- CP-001 Retention Declarations (SI-12.4, SI-13.2) ---
RETENTION_DECLARATIONS = (
    RetentionDeclaration(
        data_category="authorization_decisions",
        owner="authorization_node",
        retention_period="log_rotation_500mb",
        access_control="operator-only",
        deletion_mechanism="log_rotation",
    ),
    RetentionDeclaration(
        data_category="authorization_state_transitions",
        owner="authorization_node",
        retention_period="log_rotation_500mb",
        access_control="operator-only",
        deletion_mechanism="log_rotation",
    ),
)

# --- CP-010 Boundary Contracts (SI-11.1) ---
IDENTITY_CALLBACK_CONTRACT = BoundaryContract(
    boundary_name="authorization_identity_callback",
    boundary_type=BoundaryType.IPC,
    timeout_sec=0.01,
    error_behavior=ErrorBehavior.DEGRADE,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.INTERNAL}),
)

SIMPLE_CALLBACK_CONTRACT = BoundaryContract(
    boundary_name="authorization_simple_callback",
    boundary_type=BoundaryType.IPC,
    timeout_sec=0.005,
    error_behavior=ErrorBehavior.DEGRADE,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.INTERNAL}),
)

REQUEST_FOLLOW_CONTRACT = BoundaryContract(
    boundary_name="authorization_request_follow",
    boundary_type=BoundaryType.IPC,
    timeout_sec=5.0,
    error_behavior=ErrorBehavior.FAIL,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.INTERNAL}),
)


class AuthState(IntEnum):
    """Authorization states matching AuthorizedTarget.msg."""
    UNKNOWN = -1    # Uninitialized / unrecognized (CP-010)
    UNAUTHORIZED = 0
    ACQUIRING = 1
    AUTHORIZED = 2
    SUSPENDED = 3


ACTIVE_STATES = frozenset({AuthState.AUTHORIZED, AuthState.SUSPENDED})
TERMINAL_STATES = frozenset()


@dataclass
class FaceMatchEvidence:
    """Single face match observation for 2-of-N consistency check."""
    timestamp: float  # ROS time as float seconds
    face_track_id: int
    person_track_id: int
    user_id: str
    score: float
    margin: float
    quality_ok: bool
    association_confidence: float


@dataclass
class ConflictObservation:
    """Observation of face evidence on a different track (for persistence gate)."""
    timestamp: float
    different_person_track_id: int
    user_id: str
    association_confidence: float


@dataclass
class AuthorizationState:
    """Current authorization state."""
    state: AuthState = AuthState.UNAUTHORIZED
    authorized_track_id: int = 0
    authorized_user_id: str = ""
    confidence: float = 0.0
    last_face_evidence_time: float = 0.0
    state_entry_time: float = 0.0
    acquisition_attempts: int = 0
    refresh_count: int = 0
    reason_code: int = 0
    reason_text: str = ""

    # Last known track info (for publishing)
    track_state: int = 0
    track_match_margin: float = 0.0
    track_last_update_time: float = 0.0


class AuthorizationNode(Node, RosHealthPublisher):
    """ROS 2 node for authorization management."""

    def __init__(self):
        super().__init__("authorization_node")
        self._health_init("authorization_node")

        # Use reentrant callback group for service calls
        self.callback_group = ReentrantCallbackGroup()

        # Declare parameters
        self._declare_parameters()

        # Get parameters
        self._load_parameters()

        # State
        self.state = AuthorizationState()
        self.state_lock = Lock()

        # Evidence buffers
        self.face_match_buffer: deque = deque(maxlen=20)
        self.conflict_buffer: deque = deque(maxlen=10)
        self.last_refresh_time: float = 0.0

        # Cached inputs
        self.latest_person_tracks: Dict[int, PersonTrack] = {}
        self.latest_face_tracks: Dict[int, FaceTrack] = {}
        self.latest_identity_candidates: Dict[int, FaceIdentityCandidate] = {}

        # Acquisition waiting (for blocking service)
        self.acquisition_in_progress = False
        self.acquisition_start_time: float = 0.0
        self.acquisition_timeout_sec: float = 5.0
        self.acquisition_session_id: str = ""
        self.acquisition_turn_id: str = ""

        # QoS: Best-effort for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.person_tracks_sub = self.create_subscription(
            PersonTracks,
            "/tracking/person_tracks",
            self._person_tracks_callback,
            sensor_qos,
            callback_group=self.callback_group,
        )

        self.face_tracks_sub = self.create_subscription(
            FaceTracks,
            "/perception/faces/tracks",
            self._face_tracks_callback,
            sensor_qos,
            callback_group=self.callback_group,
        )

        self.identity_sub = self.create_subscription(
            FaceIdentityCandidates,
            "/perception/face_id/candidates",
            self._identity_callback,
            sensor_qos,
            callback_group=self.callback_group,
        )

        # Publisher
        self.auth_pub = self.create_publisher(
            AuthorizedTarget,
            "/world/authorized_target",
            sensor_qos,
        )

        # Services
        self.request_follow_srv = self.create_service(
            RequestFollow,
            "/authorization/request_follow",
            self._request_follow_callback,
            callback_group=self.callback_group,
        )

        self.cancel_follow_srv = self.create_service(
            CancelFollow,
            "/authorization/cancel_follow",
            self._cancel_follow_callback,
            callback_group=self.callback_group,
        )

        # Timers
        publish_period = 1.0 / self.publish_rate_hz
        self.publish_timer = self.create_timer(
            publish_period,
            self._publish_state,
            callback_group=self.callback_group,
        )

        # Independent liveness watchdog (CP-002, AD-068)
        self._watchdog = WatchdogThread(
            name="authorization_node",
            interval_sec=3.0,
            on_timeout=self._on_watchdog_timeout,
        )
        self._watchdog.start()

        # Degradation policy (CP-009, AD-070)
        self._deg_policy = DegradationPolicy("authorization_node", [
            DepPolicy(
                dependency="face_id_candidates",
                failure_modes=[FailureMode.UNAVAILABLE],
                response=ResponseType.DEGRADE_WITH_NOTIFICATION,
                recovery_target=RecoveryTarget.FULL_OPERATION,
                recovery_action="Check auraface_id_node; verify "
                    "/perception/face_id/candidates topic is publishing",
            ),
            DepPolicy(
                dependency="person_tracks",
                failure_modes=[FailureMode.UNAVAILABLE],
                response=ResponseType.DEGRADE_WITH_NOTIFICATION,
                recovery_target=RecoveryTarget.FULL_OPERATION,
                recovery_action="Check person_detector_node and tracker; "
                    "verify /tracking/person_tracks topic is publishing",
            ),
        ])
        self._deg_state = DegradedState(self._deg_policy)

        self.get_logger().info("Authorization node initialized")

    def _on_watchdog_timeout(self, name: str, elapsed_sec: float) -> None:
        """Watchdog timeout — force UNAUTHORIZED to block stale authorization."""
        with self.state_lock:
            if self.state.state == AuthState.UNAUTHORIZED:
                return
            now = self._get_ros_time_sec()
            self._transition_to_unauthorized(
                AuthorizedTarget.REASON_TIMEOUT,
                f"Watchdog timeout: no subscription activity for {elapsed_sec:.1f}s",
                now,
            )
            self.acquisition_in_progress = False
        _response = self._deg_state.report_degradation(
            "face_id_candidates", FailureMode.UNAVAILABLE,
            f"No face ID data for {elapsed_sec:.1f}s — forcing UNAUTHORIZED",
        )
        _response = self._deg_state.report_degradation(
            "person_tracks", FailureMode.UNAVAILABLE,
            f"No tracking data for {elapsed_sec:.1f}s",
        )
        self.get_logger().error(
            f"WATCHDOG TIMEOUT: authorization node stalled for {elapsed_sec:.1f}s, "
            "forced UNAUTHORIZED"
        )

    def _declare_parameters(self):
        """Declare all node parameters."""
        # Acquisition (2-of-N rule)
        self.declare_parameter("acquisition_timeout_sec", 5.0)
        self.declare_parameter("acquisition_window_sec", 0.5)
        self.declare_parameter("acquisition_min_consistent", 2)
        self.declare_parameter("min_face_match_score", 0.5)
        self.declare_parameter("min_face_match_margin", 0.08)
        self.declare_parameter("require_quality_ok", True)
        self.declare_parameter("time_coherence_tolerance_sec", 0.15)

        # Confidence
        self.declare_parameter("initial_confidence", 0.9)
        self.declare_parameter("decay_half_life_sec", 60.0)
        self.declare_parameter("min_confidence_threshold", 0.3)

        # Refresh
        self.declare_parameter("refresh_confidence_boost", 0.95)
        self.declare_parameter("refresh_min_interval_sec", 0.5)

        # Ambiguity thresholds
        self.declare_parameter("track_ambiguity_threshold", 0.1)  # match_margin < this = ambiguous
        self.declare_parameter("min_association_confidence", 0.3)
        self.declare_parameter("conflict_persistence_count", 2)
        self.declare_parameter("conflict_persistence_window_sec", 1.0)

        # Suspension
        self.declare_parameter("suspended_timeout_sec", 30.0)

        # Publishing
        self.declare_parameter("publish_rate_hz", 10.0)

    def _load_parameters(self):
        """Load parameters into instance variables."""
        self.acquisition_window_sec = self.get_parameter("acquisition_window_sec").value
        self.acquisition_min_consistent = self.get_parameter("acquisition_min_consistent").value
        self.min_face_match_score = self.get_parameter("min_face_match_score").value
        self.min_face_match_margin = self.get_parameter("min_face_match_margin").value
        self.require_quality_ok = self.get_parameter("require_quality_ok").value
        self.time_coherence_tolerance_sec = self.get_parameter("time_coherence_tolerance_sec").value

        self.initial_confidence = self.get_parameter("initial_confidence").value
        self.decay_half_life_sec = self.get_parameter("decay_half_life_sec").value
        self.min_confidence_threshold = self.get_parameter("min_confidence_threshold").value

        self.refresh_confidence_boost = self.get_parameter("refresh_confidence_boost").value
        self.refresh_min_interval_sec = self.get_parameter("refresh_min_interval_sec").value

        self.track_ambiguity_threshold = self.get_parameter("track_ambiguity_threshold").value
        self.min_association_confidence = self.get_parameter("min_association_confidence").value
        self.conflict_persistence_count = self.get_parameter("conflict_persistence_count").value
        self.conflict_persistence_window_sec = self.get_parameter("conflict_persistence_window_sec").value

        self.suspended_timeout_sec = self.get_parameter("suspended_timeout_sec").value
        self.publish_rate_hz = self.get_parameter("publish_rate_hz").value

    def _health_ok(self) -> bool:
        """Healthy when not degraded."""
        return not self._deg_state.is_degraded()

    def _health_status(self) -> str:
        """CP-009: report 'degraded' when dependency is degraded."""
        if self._deg_state.is_degraded():
            return "degraded"
        return "healthy"

    def _get_ros_time_sec(self) -> float:
        """Get current ROS time as float seconds."""
        now = self.get_clock().now()
        return now.nanoseconds / 1e9

    def _msg_stamp_to_sec(self, stamp) -> float:
        """Convert ROS message stamp to float seconds."""
        return stamp.sec + stamp.nanosec * 1e-9

    # -------------------------------------------------------------------------
    # Subscription callbacks
    # -------------------------------------------------------------------------

    def _person_tracks_callback(self, msg: PersonTracks):
        """Cache latest person tracks."""
        t0 = time.monotonic()
        self._watchdog.heartbeat()
        self._deg_state.report_recovery("person_tracks")
        with self.state_lock:
            self.latest_person_tracks = {t.track_id: t for t in msg.tracks}

            # Update authorized track info if we're tracking one
            if self.state.state in ACTIVE_STATES:
                track = self.latest_person_tracks.get(self.state.authorized_track_id)
                if track is not None:
                    self.state.track_state = track.state
                    self.state.track_match_margin = track.match_margin
                    self.state.track_last_update_time = self._msg_stamp_to_sec(track.stamp)

                    # Check for track LOST or ambiguity while AUTHORIZED
                    if self.state.state == AuthState.AUTHORIZED:
                        self._check_track_triggers(track)
        timing_ms = round((time.monotonic() - t0) * 1000, 1)
        if timing_ms > SIMPLE_CALLBACK_CONTRACT.timeout_sec * 1000:
            _structured_logger.warning(
                "boundary_budget_overrun",
                trace_ctx=None,
                **SIMPLE_CALLBACK_CONTRACT.log_fields(),
                timing_ms=timing_ms,
            )

    def _face_tracks_callback(self, msg: FaceTracks):
        """Cache latest face tracks for association lookup."""
        t0 = time.monotonic()
        self._watchdog.heartbeat()
        with self.state_lock:
            self.latest_face_tracks = {t.track_id: t for t in msg.tracks}
        timing_ms = round((time.monotonic() - t0) * 1000, 1)
        if timing_ms > SIMPLE_CALLBACK_CONTRACT.timeout_sec * 1000:
            _structured_logger.warning(
                "boundary_budget_overrun",
                trace_ctx=None,
                **SIMPLE_CALLBACK_CONTRACT.log_fields(),
                timing_ms=timing_ms,
            )

    def _identity_callback(self, msg: FaceIdentityCandidates):
        """Process identity candidates for acquisition/refresh/conflict detection."""
        t0 = time.monotonic()
        self._deg_state.report_recovery("face_id_candidates")
        now = self._get_ros_time_sec()
        msg_time = self._msg_stamp_to_sec(msg.header.stamp)

        with self.state_lock:
            self.latest_identity_candidates = {c.track_id: c for c in msg.candidates}

            for candidate in msg.candidates:
                self._process_identity_candidate(candidate, msg_time, now)
        timing_ms = round((time.monotonic() - t0) * 1000, 1)
        if timing_ms > IDENTITY_CALLBACK_CONTRACT.timeout_sec * 1000:
            _structured_logger.warning(
                "boundary_budget_overrun",
                trace_ctx=None,
                **IDENTITY_CALLBACK_CONTRACT.log_fields(),
                timing_ms=timing_ms,
            )

    def _process_identity_candidate(
        self,
        candidate: FaceIdentityCandidate,
        msg_time: float,
        now: float
    ):
        """Process a single identity candidate."""
        # Debug: log all incoming candidates during acquisition
        if self.state.state == AuthState.ACQUIRING:
            self.get_logger().info(
                f"[ACQUIRE] Candidate: track_id={candidate.track_id} "
                f"user_id={candidate.user_id} score={candidate.score:.3f} margin={candidate.margin:.3f}"
            )

        # Get associated face track
        face_track = self.latest_face_tracks.get(candidate.track_id)
        if face_track is None:
            if self.state.state == AuthState.ACQUIRING:
                self.get_logger().warn(
                    f"[ACQUIRE] DROPPED: face_track_id={candidate.track_id} not in cache "
                    f"(cache has {len(self.latest_face_tracks)} tracks: {list(self.latest_face_tracks.keys())})"
                )
            return

        person_track_id = face_track.associated_person_track_id
        if person_track_id == 0:
            if self.state.state == AuthState.ACQUIRING:
                self.get_logger().warn(
                    f"[ACQUIRE] DROPPED: face_track_id={candidate.track_id} has no person association"
                )
            return

        # Skip time coherence check for dev (face_track.last_seen uses different clock domain)
        # TODO: Fix timestamp domains to use consistent clock source
        # face_time = self._msg_stamp_to_sec(face_track.last_seen)
        # time_diff = abs(face_time - msg_time)
        # if time_diff > self.time_coherence_tolerance_sec:
        #     if self.state.state == AuthState.ACQUIRING:
        #         self.get_logger().warn(...)
        #     return

        # Build evidence record
        evidence = FaceMatchEvidence(
            timestamp=msg_time,
            face_track_id=candidate.track_id,
            person_track_id=person_track_id,
            user_id=candidate.user_id,
            score=candidate.score,
            margin=candidate.margin,
            quality_ok=face_track.quality_ok,
            association_confidence=face_track.association_confidence,
        )

        # Route based on current state
        if self.state.state == AuthState.ACQUIRING:
            self._handle_acquiring_evidence(evidence, now)
        elif self.state.state == AuthState.AUTHORIZED:
            self._handle_authorized_evidence(evidence, now)
        elif self.state.state == AuthState.SUSPENDED:
            self._handle_suspended_evidence(evidence, now)

    # -------------------------------------------------------------------------
    # State-specific evidence handling
    # -------------------------------------------------------------------------

    def _handle_acquiring_evidence(self, evidence: FaceMatchEvidence, now: float):
        """Handle evidence during ACQUIRING state."""
        # Check if evidence meets threshold
        if not self._evidence_meets_threshold(evidence, log_failures=True):
            return

        # Add to buffer - log this as it's a valid evidence
        self.get_logger().info(
            f"[ACQUIRE] VALID evidence: user={evidence.user_id} person_track={evidence.person_track_id} "
            f"score={evidence.score:.3f} margin={evidence.margin:.3f} assoc_conf={evidence.association_confidence:.3f}"
        )
        self.face_match_buffer.append(evidence)

        # Try 2-of-N consistency check
        result = self._check_2_of_n_consistency(now)
        if result is not None:
            person_track_id, user_id, avg_score = result
            self._transition_to_authorized(person_track_id, user_id, avg_score, now)

    def _handle_authorized_evidence(self, evidence: FaceMatchEvidence, now: float):
        """Handle evidence during AUTHORIZED state."""
        # Check if evidence is for the authorized track
        if evidence.person_track_id == self.state.authorized_track_id:
            # Potential refresh
            if self._evidence_meets_threshold(evidence):
                if evidence.user_id == self.state.authorized_user_id:
                    self._try_refresh(evidence, now)
                else:
                    # Different user on same track - log warning
                    self.get_logger().warn(
                        f"Different user {evidence.user_id} on authorized track "
                        f"{self.state.authorized_track_id}"
                    )
        else:
            # Evidence on different track - check for conflict
            if self._evidence_meets_threshold(evidence):
                if evidence.user_id == self.state.authorized_user_id:
                    self._record_conflict(evidence, now)

    def _handle_suspended_evidence(self, evidence: FaceMatchEvidence, now: float):
        """Handle evidence during SUSPENDED state (re-acquisition)."""
        # Must match the same user for reacquisition
        if evidence.user_id != self.state.authorized_user_id:
            return

        if not self._evidence_meets_threshold(evidence):
            return

        # Add to buffer
        self.face_match_buffer.append(evidence)

        # Try 2-of-N consistency check
        result = self._check_2_of_n_consistency(now, user_filter=self.state.authorized_user_id)
        if result is not None:
            person_track_id, user_id, avg_score = result
            # May have new track_id if old track was LOST
            self._transition_to_authorized(
                person_track_id,
                user_id,
                avg_score,
                now,
                is_reacquisition=True
            )

    def _evidence_meets_threshold(self, evidence: FaceMatchEvidence, log_failures: bool = False) -> bool:
        """Check if evidence meets quality and score thresholds."""
        if self.require_quality_ok and not evidence.quality_ok:
            if log_failures:
                self.get_logger().warn(
                    f"[ACQUIRE] THRESHOLD FAIL: quality_ok={evidence.quality_ok} (required=True)"
                )
            return False
        if evidence.score < self.min_face_match_score:
            if log_failures:
                self.get_logger().warn(
                    f"[ACQUIRE] THRESHOLD FAIL: score={evidence.score:.3f} < min={self.min_face_match_score}"
                )
            return False
        if evidence.margin < self.min_face_match_margin:
            if log_failures:
                self.get_logger().warn(
                    f"[ACQUIRE] THRESHOLD FAIL: margin={evidence.margin:.3f} < min={self.min_face_match_margin}"
                )
            return False
        if evidence.association_confidence < self.min_association_confidence:
            if log_failures:
                self.get_logger().warn(
                    f"[ACQUIRE] THRESHOLD FAIL: association_confidence={evidence.association_confidence:.3f} < min={self.min_association_confidence}"
                )
            return False
        return True

    def _check_2_of_n_consistency(
        self,
        now: float,
        user_filter: Optional[str] = None
    ) -> Optional[Tuple[int, str, float]]:
        """Check for 2 consistent matches within window.

        Returns (person_track_id, user_id, avg_score) if consistent, None otherwise.
        Also checks for multi-user ambiguity.
        """
        window_start = now - self.acquisition_window_sec

        # Filter recent evidence
        recent = [
            e for e in self.face_match_buffer
            if e.timestamp >= window_start
        ]

        if len(recent) < self.acquisition_min_consistent:
            return None

        # Group by (person_track_id, user_id)
        groups: Dict[Tuple[int, str], List[FaceMatchEvidence]] = {}
        for e in recent:
            if user_filter is not None and e.user_id != user_filter:
                continue
            key = (e.person_track_id, e.user_id)
            if key not in groups:
                groups[key] = []
            groups[key].append(e)

        # Find groups with enough consistent matches
        valid_groups = [
            (key, evs) for key, evs in groups.items()
            if len(evs) >= self.acquisition_min_consistent
        ]

        if not valid_groups:
            return None

        # Check for multi-user ambiguity (AMBIGUOUS_USERS)
        if len(valid_groups) > 1:
            # Multiple valid groups - check if different users
            users = set(key[1] for key, _ in valid_groups)
            if len(users) > 1:
                # Multiple users within margin - ambiguous
                self.state.reason_code = AuthorizedTarget.REASON_AMBIGUOUS_USERS
                self.state.reason_text = f"Multiple users matched: {', '.join(users)}"
                return None

        # Return the best group (most evidence, then highest avg score)
        best_key, best_evs = max(
            valid_groups,
            key=lambda x: (len(x[1]), sum(e.score for e in x[1]) / len(x[1]))
        )

        avg_score = sum(e.score for e in best_evs) / len(best_evs)
        return (best_key[0], best_key[1], avg_score)

    # -------------------------------------------------------------------------
    # State transitions
    # -------------------------------------------------------------------------

    def _transition_to_authorized(
        self,
        person_track_id: int,
        user_id: str,
        confidence: float,
        now: float,
        is_reacquisition: bool = False
    ):
        """Transition to AUTHORIZED state."""
        old_state = self.state.state

        self.state.state = AuthState.AUTHORIZED
        self.state.authorized_track_id = person_track_id
        self.state.authorized_user_id = user_id
        self.state.confidence = min(confidence, self.initial_confidence)
        self.state.last_face_evidence_time = now
        self.state.state_entry_time = now
        self.state.refresh_count = 0

        if is_reacquisition:
            self.state.reason_code = AuthorizedTarget.REASON_REACQUIRED_FACE_MATCH
            self.state.reason_text = f"Reacquired {user_id} on track {person_track_id}"
        else:
            self.state.reason_code = AuthorizedTarget.REASON_ACQUIRED_FACE_MATCH
            self.state.reason_text = f"Acquired {user_id} on track {person_track_id}"

        # Clear buffers
        self.face_match_buffer.clear()
        self.conflict_buffer.clear()

        # Update track info
        track = self.latest_person_tracks.get(person_track_id)
        if track is not None:
            self.state.track_state = track.state
            self.state.track_match_margin = track.match_margin
            self.state.track_last_update_time = self._msg_stamp_to_sec(track.stamp)

        self.get_logger().info(
            f"AUTHORIZED: {user_id} on track {person_track_id} "
            f"(conf={confidence:.2f}, reacq={is_reacquisition})"
        )

        # CP-001: Decision record for authorization (SI-10.1, AD-072)
        _structured_logger.decision_record(
            "SI-10.1",
            "authorized",
            {
                "user_id": user_id,
                "person_track_id": person_track_id,
                "confidence": round(confidence, 3),
                "is_reacquisition": is_reacquisition,
                "prior_state": old_state.name,
            },
        )

        # Signal acquisition complete
        self.acquisition_in_progress = False

    def _transition_to_suspended(self, reason_code: int, reason_text: str, now: float):
        """Transition to SUSPENDED state."""
        self.state.state = AuthState.SUSPENDED
        self.state.state_entry_time = now
        self.state.reason_code = reason_code
        self.state.reason_text = reason_text

        # Clear buffers for reacquisition
        self.face_match_buffer.clear()
        self.conflict_buffer.clear()

        self.get_logger().warn(
            f"SUSPENDED: {reason_text} (track={self.state.authorized_track_id})"
        )

        # CP-001: Decision record for suspension (SI-10.1, AD-072)
        _structured_logger.decision_record(
            "SI-10.1",
            "suspended",
            {
                "reason_code": reason_code,
                "authorized_track_id": self.state.authorized_track_id,
                "authorized_user_id": self.state.authorized_user_id,
            },
        )

    def _transition_to_unauthorized(self, reason_code: int, reason_text: str, now: float):
        """Transition to UNAUTHORIZED state."""
        old_track = self.state.authorized_track_id
        old_user = self.state.authorized_user_id

        self.state.state = AuthState.UNAUTHORIZED
        self.state.authorized_track_id = 0
        self.state.authorized_user_id = ""
        self.state.confidence = 0.0
        self.state.last_face_evidence_time = 0.0
        self.state.state_entry_time = now
        self.state.acquisition_attempts = 0
        self.state.refresh_count = 0
        self.state.reason_code = reason_code
        self.state.reason_text = reason_text

        # Clear track info
        self.state.track_state = 0
        self.state.track_match_margin = 0.0
        self.state.track_last_update_time = 0.0

        # Clear buffers
        self.face_match_buffer.clear()
        self.conflict_buffer.clear()

        self.get_logger().info(f"UNAUTHORIZED: {reason_text}")

        # CP-001: Decision record for deauthorization (SI-10.1, AD-072)
        _structured_logger.decision_record(
            "SI-10.1",
            "deauthorized",
            {
                "reason_code": reason_code,
                "prior_track_id": old_track,
                "prior_user_id": old_user,
            },
        )

        return old_track, old_user

    # -------------------------------------------------------------------------
    # Trigger checks
    # -------------------------------------------------------------------------

    def _check_track_triggers(self, track: PersonTrack):
        """Check for triggers that should move AUTHORIZED -> SUSPENDED."""
        now = self._get_ros_time_sec()

        # 1. Track LOST
        if track.state == PersonTrack.STATE_LOST:
            self._transition_to_suspended(
                AuthorizedTarget.REASON_TARGET_LOST,
                f"Track {track.track_id} entered LOST state",
                now
            )
            return

        # 2. Track ambiguous (match_margin < threshold)
        if track.match_margin < self.track_ambiguity_threshold:
            self._transition_to_suspended(
                AuthorizedTarget.REASON_TRACK_AMBIGUOUS,
                f"Track {track.track_id} ambiguous (margin={track.match_margin:.3f})",
                now
            )
            return

    def _record_conflict(self, evidence: FaceMatchEvidence, now: float):
        """Record conflict observation (authorized user on different track)."""
        obs = ConflictObservation(
            timestamp=now,
            different_person_track_id=evidence.person_track_id,
            user_id=evidence.user_id,
            association_confidence=evidence.association_confidence,
        )
        self.conflict_buffer.append(obs)

        # Check persistence gate (K of M)
        self._check_conflict_persistence(now)

    def _check_conflict_persistence(self, now: float):
        """Check if conflicts meet persistence gate (2-of-3 within 1s)."""
        window_start = now - self.conflict_persistence_window_sec

        recent = [
            o for o in self.conflict_buffer
            if o.timestamp >= window_start
        ]

        if len(recent) >= self.conflict_persistence_count:
            # Conflict persisted - transition to SUSPENDED
            self._transition_to_suspended(
                AuthorizedTarget.REASON_AMBIGUOUS_MULTIPLE,
                f"Authorized user detected on different track "
                f"({self.conflict_persistence_count} observations)",
                now
            )

    def _try_refresh(self, evidence: FaceMatchEvidence, now: float):
        """Try to refresh confidence with new face evidence."""
        # Rate limit
        if now - self.last_refresh_time < self.refresh_min_interval_sec:
            return

        self.state.confidence = self.refresh_confidence_boost
        self.state.last_face_evidence_time = now
        self.state.refresh_count += 1
        self.last_refresh_time = now

        self.get_logger().debug(
            f"Refresh #{self.state.refresh_count}: confidence reset to {self.refresh_confidence_boost}"
        )

    def _update_confidence_decay(self, now: float):
        """Apply confidence decay based on time since last face evidence."""
        if self.state.state != AuthState.AUTHORIZED:
            return

        if self.state.last_face_evidence_time <= 0:
            return

        elapsed = now - self.state.last_face_evidence_time
        if elapsed <= 0:
            return

        # Exponential decay: conf = initial * 0.5^(elapsed / half_life)
        decay_factor = math.pow(0.5, elapsed / self.decay_half_life_sec)
        self.state.confidence = self.initial_confidence * decay_factor

        # Check threshold
        if self.state.confidence < self.min_confidence_threshold:
            self._transition_to_suspended(
                AuthorizedTarget.REASON_CONFIDENCE_DECAY,
                f"Confidence decayed to {self.state.confidence:.3f}",
                now
            )

    def _check_timeouts(self, now: float):
        """Check for state timeouts."""
        if self.state.state == AuthState.ACQUIRING:
            elapsed = now - self.state.state_entry_time
            if elapsed >= self.acquisition_timeout_sec:
                self.state.acquisition_attempts += 1
                self._transition_to_unauthorized(
                    AuthorizedTarget.REASON_TIMEOUT,
                    f"Acquisition timeout after {elapsed:.1f}s",
                    now
                )
                self.acquisition_in_progress = False

        elif self.state.state == AuthState.SUSPENDED:
            elapsed = now - self.state.state_entry_time
            if elapsed >= self.suspended_timeout_sec:
                self._transition_to_unauthorized(
                    AuthorizedTarget.REASON_TIMEOUT,
                    f"Suspension timeout after {elapsed:.1f}s",
                    now
                )

    # -------------------------------------------------------------------------
    # Service callbacks
    # -------------------------------------------------------------------------

    def _request_follow_callback(
        self,
        request: RequestFollow.Request,
        response: RequestFollow.Response
    ) -> RequestFollow.Response:
        """Handle RequestFollow service (blocking)."""
        now = self._get_ros_time_sec()
        start_time = now

        timeout_sec = request.timeout_ms / 1000.0 if request.timeout_ms > 0 else 5.0
        t0 = time.monotonic()

        with self.state_lock:
            if self.state.state == AuthState.AUTHORIZED:
                # Already authorized
                response.success = True
                response.error_code = "OK"
                response.authorized_track_id = self.state.authorized_track_id
                response.authorized_user_id = self.state.authorized_user_id
                response.confidence = self.state.confidence
                response.latency_ms = 0
                response.reason = "Already authorized"
                return response

            if self.state.state == AuthState.ACQUIRING:
                # Already acquiring - could wait or return error
                response.success = False
                response.error_code = "ALREADY_ACQUIRING"
                response.reason = "Acquisition already in progress"
                return response

            # Start acquisition
            self.state.state = AuthState.ACQUIRING
            self.state.state_entry_time = now
            self.state.reason_code = AuthorizedTarget.REASON_NONE
            self.state.reason_text = "Acquiring..."
            self.acquisition_in_progress = True
            self.acquisition_start_time = now
            self.acquisition_timeout_sec = timeout_sec
            self.acquisition_session_id = request.session_id
            self.acquisition_turn_id = request.turn_id

            self.face_match_buffer.clear()

        self.get_logger().info(
            f"RequestFollow: starting acquisition (timeout={timeout_sec}s)"
        )

        # Poll for completion (blocking)
        poll_interval = 0.05  # 50ms
        while True:
            time.sleep(poll_interval)
            now = self._get_ros_time_sec()

            with self.state_lock:
                if self.state.state == AuthState.AUTHORIZED:
                    # Success
                    response.success = True
                    response.error_code = "OK"
                    response.authorized_track_id = self.state.authorized_track_id
                    response.authorized_user_id = self.state.authorized_user_id
                    response.confidence = self.state.confidence
                    response.latency_ms = int((now - start_time) * 1000)
                    response.reason = self.state.reason_text
                    timing_ms = round((time.monotonic() - t0) * 1000, 1)
                    if timing_ms > REQUEST_FOLLOW_CONTRACT.timeout_sec * 1000:
                        _structured_logger.warning(
                            "boundary_budget_overrun",
                            trace_ctx=None,
                            **REQUEST_FOLLOW_CONTRACT.log_fields(),
                            timing_ms=timing_ms,
                        )
                    return response

                if self.state.state == AuthState.UNAUTHORIZED:
                    # Failed (timeout or cancelled)
                    response.success = False
                    if self.state.reason_code == AuthorizedTarget.REASON_TIMEOUT:
                        response.error_code = "TIMEOUT"
                    elif self.state.reason_code == AuthorizedTarget.REASON_AMBIGUOUS_USERS:
                        response.error_code = "AMBIGUOUS_USERS"
                    else:
                        response.error_code = "NO_MATCH"
                    response.latency_ms = int((now - start_time) * 1000)
                    response.reason = self.state.reason_text
                    timing_ms = round((time.monotonic() - t0) * 1000, 1)
                    if timing_ms > REQUEST_FOLLOW_CONTRACT.timeout_sec * 1000:
                        _structured_logger.warning(
                            "boundary_budget_overrun",
                            trace_ctx=None,
                            **REQUEST_FOLLOW_CONTRACT.log_fields(),
                            timing_ms=timing_ms,
                        )
                    return response

                # Still acquiring - check timeout
                elapsed = now - start_time
                if elapsed >= timeout_sec:
                    self.state.acquisition_attempts += 1
                    self._transition_to_unauthorized(
                        AuthorizedTarget.REASON_TIMEOUT,
                        f"Acquisition timeout after {elapsed:.1f}s",
                        now
                    )
                    self.acquisition_in_progress = False

                    response.success = False
                    response.error_code = "TIMEOUT"
                    response.latency_ms = int(elapsed * 1000)
                    response.reason = "No face match within timeout"
                    timing_ms = round((time.monotonic() - t0) * 1000, 1)
                    if timing_ms > REQUEST_FOLLOW_CONTRACT.timeout_sec * 1000:
                        _structured_logger.warning(
                            "boundary_budget_overrun",
                            trace_ctx=None,
                            **REQUEST_FOLLOW_CONTRACT.log_fields(),
                            timing_ms=timing_ms,
                        )
                    return response

    def _cancel_follow_callback(
        self,
        request: CancelFollow.Request,
        response: CancelFollow.Response
    ) -> CancelFollow.Response:
        """Handle CancelFollow service."""
        now = self._get_ros_time_sec()

        with self.state_lock:
            if self.state.state == AuthState.UNAUTHORIZED:
                response.success = True
                response.error_code = "NOT_AUTHORIZED"
                response.previous_track_id = 0
                response.previous_user_id = ""
                return response

            prev_track, prev_user = self._transition_to_unauthorized(
                AuthorizedTarget.REASON_CANCELLED,
                "Cancelled via CancelFollow service",
                now
            )
            self.acquisition_in_progress = False

            response.success = True
            response.error_code = "OK"
            response.previous_track_id = prev_track
            response.previous_user_id = prev_user

        self.get_logger().info(
            f"CancelFollow: cancelled authorization for {prev_user} on track {prev_track}"
        )

        return response

    # -------------------------------------------------------------------------
    # Publishing
    # -------------------------------------------------------------------------

    def _publish_state(self):
        """Publish current authorization state."""
        self._watchdog.heartbeat()
        now = self._get_ros_time_sec()

        with self.state_lock:
            # Update decay and check timeouts
            self._update_confidence_decay(now)
            self._check_timeouts(now)

            # Build message
            msg = AuthorizedTarget()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "authorization"

            msg.auth_state = self.state.state
            msg.authorized_track_id = self.state.authorized_track_id
            msg.authorized_user_id = self.state.authorized_user_id
            msg.confidence = self.state.confidence

            if self.state.last_face_evidence_time > 0:
                msg.last_face_evidence_age_ms = max(0, int(
                    (now - self.state.last_face_evidence_time) * 1000
                ))
            else:
                msg.last_face_evidence_age_ms = 0

            msg.reason_code = self.state.reason_code
            msg.reason_text = self.state.reason_text
            msg.track_state = self.state.track_state
            msg.track_match_margin = self.state.track_match_margin

            if self.state.track_last_update_time > 0:
                msg.authorized_track_age_ms = max(0, int(
                    (now - self.state.track_last_update_time) * 1000
                ))
            else:
                msg.authorized_track_age_ms = 0

            msg.time_in_state_ms = max(0, int((now - self.state.state_entry_time) * 1000))
            msg.acquisition_attempts = self.state.acquisition_attempts
            msg.refresh_count = self.state.refresh_count

        self.auth_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AuthorizationNode()

    # Use MultiThreadedExecutor so subscription callbacks can run
    # while service callback is blocked waiting for acquisition
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._watchdog.stop()
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
