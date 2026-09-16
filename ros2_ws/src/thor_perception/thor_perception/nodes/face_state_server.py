#!/usr/bin/env python3
"""Face State Server - aggregates face tracks and provides query services.

This node subscribes to /perception/faces/tracks, maintains time-windowed state,
applies temporal quality gating (TOO_NEW flag), and exposes the GetFaceSnapshot service.

Responsibilities:
- Aggregate face detections from face_detection_node
- Maintain rolling history per track
- Apply quality_min_time_in_track_ms before setting quality_ok
- Expose /perception/faces/get_snapshot service
- Detect tracker reset conditions (ID churn, timestamp jumps)
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time as RclpyTime
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point32
from vision_msgs.msg import BoundingBox2D

from thor_msgs.msg import FaceTracks, FaceTrack, FaceLandmarks
from thor_msgs.srv import GetFaceSnapshot
from thor_behavior import (
    WatchdogThread,
    DegradationPolicy, DegradedState, DepPolicy, FailureMode, RecoveryTarget,
    ResponseType,
)
from thor_telemetry import get_logger as _get_structured_logger, ErrorCode
from thor_telemetry import BoundaryContract, BoundaryType, ErrorBehavior
from thor_perception.health.ros_health_publisher import RosHealthPublisher

_structured_logger = _get_structured_logger("face_state_server")

_deg_policy = DegradationPolicy("face_state_server", [
    DepPolicy(
        dependency="face_tracks_topic",
        failure_modes=[FailureMode.DATA_STALE, FailureMode.TIMEOUT],
        response=ResponseType.DEGRADE_WITH_NOTIFICATION,
        recovery_target=RecoveryTarget.FULL_OPERATION,
        recovery_action="Wait for face detection pipeline to publish on /perception/faces/tracks",
    ),
])
_deg_state = DegradedState(_deg_policy, logger=_structured_logger)

# --- CP-010 Boundary Contracts (SI-11.1) ---
FACE_SNAPSHOT_CONTRACT = BoundaryContract(
    boundary_name="face_get_snapshot",
    boundary_type=BoundaryType.IPC,
    timeout_sec=1.0,
    error_behavior=ErrorBehavior.FAIL,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.INTERNAL}),
)

ON_FACE_TRACKS_CONTRACT = BoundaryContract(
    boundary_name="face_on_face_tracks",
    boundary_type=BoundaryType.IPC,
    timeout_sec=0.033,
    error_behavior=ErrorBehavior.DEGRADE,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.INTERNAL}),
)


@dataclass
class FaceSample:
    """Single face detection sample for rolling window."""
    timestamp: float
    center_x: float
    center_y: float
    confidence: float
    face_size_px: float
    bbox: BoundingBox2D
    landmarks: FaceLandmarks
    quality_flags: int
    bearing_deg: float


@dataclass
class FaceTrackState:
    """Internal state for a tracked face."""
    track_id: int
    first_seen: float
    last_seen: float
    detection_count: int = 0
    samples: deque = field(default_factory=lambda: deque(maxlen=500))

    @property
    def latest_sample(self) -> Optional[FaceSample]:
        return self.samples[-1] if self.samples else None

    @property
    def latest_confidence(self) -> float:
        return self.samples[-1].confidence if self.samples else 0.0

    @property
    def latest_face_size(self) -> float:
        return self.samples[-1].face_size_px if self.samples else 0.0

    @property
    def age_ms(self) -> int:
        """Time since first seen in milliseconds."""
        if not self.samples:
            return 0
        return int((self.last_seen - self.first_seen) * 1000)

    def average_confidence(self) -> float:
        """Compute average confidence over samples."""
        if not self.samples:
            return 0.0
        return sum(s.confidence for s in self.samples) / len(self.samples)


class FaceStateServer(Node, RosHealthPublisher):
    """Aggregates face tracks and provides GetFaceSnapshot service."""

    def __init__(self):
        super().__init__('face_state_server')
        self._health_init('face_state_server')

        # Parameters
        self.declare_parameter('retention_sec', 30.0)
        self.declare_parameter('cleanup_interval_sec', 10.0)
        self.declare_parameter('quality_min_time_in_track_ms', 200)
        self.declare_parameter('timestamp_discontinuity_sec', 2.0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)

        # Cache parameter values
        self.retention_sec = self.get_parameter('retention_sec').value
        self.cleanup_interval_sec = self.get_parameter('cleanup_interval_sec').value
        self.quality_min_time_in_track_ms = self.get_parameter('quality_min_time_in_track_ms').value
        self.timestamp_discontinuity_sec = self.get_parameter('timestamp_discontinuity_sec').value
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value

        # State storage
        self._tracks: dict[int, FaceTrackState] = {}
        self._lock = threading.Lock()
        self._latest_fps: float = 0.0
        self._total_faces_detected: int = 0
        self._frame_times: deque[float] = deque(maxlen=100)

        # Timestamp discontinuity detection
        self._last_msg_time: Optional[float] = None

        # Callback group for concurrent service handling
        self._cb_group = ReentrantCallbackGroup()

        # QoS for subscribing to face tracks (BEST_EFFORT for sensor data)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscriber
        self._tracks_sub = self.create_subscription(
            FaceTracks,
            '/perception/faces/tracks',
            self._on_face_tracks,
            sensor_qos,
            callback_group=self._cb_group,
        )

        # Service
        self._snapshot_srv = self.create_service(
            GetFaceSnapshot,
            '/perception/faces/get_snapshot',
            self._handle_get_snapshot,
            callback_group=self._cb_group,
        )

        # Cleanup timer
        self._cleanup_timer = self.create_timer(
            self.cleanup_interval_sec,
            self._cleanup_old_tracks,
            callback_group=self._cb_group,
        )

        # Watchdog: independent liveness check (CP-002)
        self._watchdog = WatchdogThread(
            name="face_state_server",
            interval_sec=5.0,
            on_timeout=self._on_watchdog_timeout,
        )
        self._stale = False
        self._watchdog.start()

        self.get_logger().info(
            f"Face state server initialized: retention={self.retention_sec}s, "
            f"min_track_time={self.quality_min_time_in_track_ms}ms"
        )

    def _on_watchdog_timeout(self, name: str, elapsed_sec: float) -> None:
        """Watchdog timeout: mark data as stale."""
        self.get_logger().warning(
            f"Watchdog timeout: no face tracks for {elapsed_sec:.1f}s — data stale"
        )
        _structured_logger.emit_failure(
                trace_ctx=None,  # CP-006 §5.1: perception has no turn-level trace
                operation="face_track_pipeline",
            error_code=ErrorCode.TIMEOUT,
            error_detail=f"No face tracks for {elapsed_sec:.1f}s",
            trigger="watchdog_timeout",
            elapsed_sec=round(elapsed_sec, 1),
        )
        self._stale = True
        _response = _deg_state.report_degradation(
            "face_tracks_topic", FailureMode.DATA_STALE,
            f"No face tracks for {elapsed_sec:.1f}s",
        )

    def _health_ok(self) -> bool:
        """Healthy if watchdog hasn't flagged stale."""
        return not self._stale

    def _health_status(self) -> str:
        """CP-009: report 'degraded' when dependency is degraded."""
        if _deg_state.is_degraded():
            return "degraded"
        return "healthy" if self._health_ok() else "unhealthy"

    def _on_face_tracks(self, msg: FaceTracks):
        """Handle incoming face tracks message."""
        t0 = time.monotonic()
        self._watchdog.heartbeat()
        self._stale = False
        _deg_state.report_recovery("face_tracks_topic")
        now = time.time()

        with self._lock:
            # Check for timestamp discontinuity before processing
            if self._check_timestamp_discontinuity_locked(now):
                return  # State was cleared, skip this message

            self._frame_times.append(now)
            self._latest_fps = msg.fps
            self._total_faces_detected = msg.total_faces_detected
            self._last_msg_time = now

            for track in msg.tracks:
                track_id = track.track_id

                # Get or create track state
                if track_id not in self._tracks:
                    self._tracks[track_id] = FaceTrackState(
                        track_id=track_id,
                        first_seen=now,
                        last_seen=now,
                    )

                state = self._tracks[track_id]
                state.last_seen = now
                state.detection_count += 1

                # Create sample
                sample = FaceSample(
                    timestamp=now,
                    center_x=track.bbox.center.position.x,
                    center_y=track.bbox.center.position.y,
                    confidence=track.detector_confidence,
                    face_size_px=track.face_size_px,
                    bbox=track.bbox,
                    landmarks=track.landmarks,
                    quality_flags=track.quality_flags,
                    bearing_deg=track.bearing_deg,
                )
                state.samples.append(sample)
        timing_ms = round((time.monotonic() - t0) * 1000, 1)
        if timing_ms > ON_FACE_TRACKS_CONTRACT.timeout_sec * 1000:
            _structured_logger.warning(
                "boundary_budget_overrun",
                trace_ctx=None,
                **ON_FACE_TRACKS_CONTRACT.log_fields(),
                timing_ms=timing_ms,
            )

    def _check_timestamp_discontinuity_locked(self, now: float) -> bool:
        """Check for timestamp discontinuity and clear state if detected.

        MUST be called with self._lock held.

        Clears track history on:
        - Timestamp going backwards
        - Timestamp jumping forward more than threshold (default 2s)

        This keeps the state server in sync with the tracker which also
        clears on discontinuity.

        Returns:
            True if discontinuity detected and state was cleared
        """
        if self._last_msg_time is None:
            return False

        dt = now - self._last_msg_time

        if dt < 0 or dt > self.timestamp_discontinuity_sec:
            self.get_logger().warn(
                f"Timestamp discontinuity detected ({dt:.2f}s), clearing track history"
            )
            self._tracks.clear()
            self._frame_times.clear()
            self._last_msg_time = now
            return True

        return False

    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been seen recently."""
        cutoff = time.time() - self.retention_sec
        with self._lock:
            stale = [tid for tid, state in self._tracks.items()
                     if state.last_seen < cutoff]
            for tid in stale:
                del self._tracks[tid]

            if stale:
                self.get_logger().debug(f"Cleaned up {len(stale)} stale face tracks")

    def _apply_temporal_quality(self, track: FaceTrackState, now: float) -> tuple[bool, str, int]:
        """Apply temporal quality gating (TOO_NEW flag).

        Returns: (quality_ok, quality_reason, quality_flags)
        """
        latest = track.latest_sample
        if not latest:
            return False, "no samples", FaceTrack.TOO_NEW

        # Start with detector's quality flags
        flags = latest.quality_flags
        reasons = []

        # Check if track is too young
        age_ms = track.age_ms
        if age_ms < self.quality_min_time_in_track_ms:
            flags |= FaceTrack.TOO_NEW
            reasons.append(f"age={age_ms}ms<{self.quality_min_time_in_track_ms}ms")

        # Parse existing reasons from detector
        if flags & FaceTrack.LOW_CONFIDENCE:
            reasons.append("low_conf")
        if flags & FaceTrack.TOO_SMALL:
            reasons.append("too_small")
        if flags & FaceTrack.BLUR:
            reasons.append("blur")
        if flags & FaceTrack.EXTREME_POSE:
            reasons.append("extreme_pose")

        quality_ok = flags == 0
        quality_reason = "; ".join(reasons) if reasons else "ok"

        return quality_ok, quality_reason, flags

    def _track_to_msg(self, state: FaceTrackState, now: float) -> FaceTrack:
        """Convert internal track state to FaceTrack message."""
        latest = state.latest_sample
        msg = FaceTrack()

        msg.track_id = state.track_id

        # Timestamps
        msg.first_seen.sec = int(state.first_seen)
        msg.first_seen.nanosec = int((state.first_seen % 1) * 1e9)
        msg.last_seen.sec = int(state.last_seen)
        msg.last_seen.nanosec = int((state.last_seen % 1) * 1e9)

        if latest:
            msg.bbox = latest.bbox
            msg.landmarks = latest.landmarks
            msg.detector_confidence = latest.confidence
            msg.face_size_px = latest.face_size_px
            msg.bearing_deg = latest.bearing_deg

        # Apply temporal quality gating
        quality_ok, quality_reason, quality_flags = self._apply_temporal_quality(state, now)
        msg.quality_ok = quality_ok
        msg.quality_reason = quality_reason
        msg.quality_flags = quality_flags

        # Tracking metrics
        msg.detection_count = state.detection_count
        msg.age_ms = state.age_ms
        msg.last_seen_ms = int((now - state.last_seen) * 1000)
        msg.stability_score = state.average_confidence()

        # Source frame (from latest sample or defaults)
        msg.source_frame_id = "camera0"
        msg.source_width = self.frame_width
        msg.source_height = self.frame_height

        # Association (unpopulated in Phase 1)
        msg.associated_person_track_id = 0
        msg.association_confidence = 0.0

        return msg

    def _handle_get_snapshot(self, request: GetFaceSnapshot.Request,
                              response: GetFaceSnapshot.Response) -> GetFaceSnapshot.Response:
        """Handle GetFaceSnapshot service request."""
        t0 = time.monotonic()
        now = time.time()

        # Clamp window to retention
        window_sec = min(request.max_age_sec, self.retention_sec) if request.max_age_sec > 0 else self.retention_sec
        cutoff = now - window_sec

        with self._lock:
            # Filter tracks by window
            active_tracks = [
                state for state in self._tracks.values()
                if state.last_seen > cutoff
            ]

            # If latest_only, only keep tracks with samples in this frame
            if request.latest_only:
                # Use most recent frame time as reference
                if self._frame_times:
                    latest_frame = self._frame_times[-1]
                    active_tracks = [
                        state for state in active_tracks
                        if abs(state.last_seen - latest_frame) < 0.1  # 100ms tolerance
                    ]

            # Convert to messages
            faces = []
            for state in active_tracks:
                track_msg = self._track_to_msg(state, now)

                # Apply quality gate filter if requested
                if request.quality_gated and not track_msg.quality_ok:
                    continue

                faces.append(track_msg)

            # Apply max_tracks limit
            if request.max_tracks > 0 and len(faces) > request.max_tracks:
                # Sort by confidence descending and take top N
                faces.sort(key=lambda f: f.detector_confidence, reverse=True)
                faces = faces[:request.max_tracks]

            # Count frames in window
            frames_in_window = sum(1 for t in self._frame_times if t > cutoff)

        response.success = True
        response.error = ""
        # CP-003 (SI-5.3): timestamp reflects newest sample time, not current wall clock.
        # Clients compute data age as (now - response.timestamp).
        if self._frame_times:
            newest_ts = self._frame_times[-1]
            ts_msg = TimeMsg()
            ts_msg.sec = int(newest_ts)
            ts_msg.nanosec = int((newest_ts - int(newest_ts)) * 1e9)
            response.timestamp = ts_msg
        else:
            response.timestamp = self.get_clock().now().to_msg()
        response.faces = faces
        response.total_frames_in_window = frames_in_window
        response.window_used_sec = window_sec

        timing_ms = round((time.monotonic() - t0) * 1000, 1)
        if timing_ms > FACE_SNAPSHOT_CONTRACT.timeout_sec * 1000:
            _structured_logger.warning(
                "boundary_budget_overrun",
                trace_ctx=None,
                **FACE_SNAPSHOT_CONTRACT.log_fields(),
                timing_ms=timing_ms,
            )
        return response


def main(args=None):
    rclpy.init(args=args)

    node = FaceStateServer()

    # Use multi-threaded executor for concurrent callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._watchdog.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
