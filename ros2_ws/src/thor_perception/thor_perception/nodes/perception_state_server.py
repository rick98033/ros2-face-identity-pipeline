"""Perception State Server - aggregates detections and provides query services.

This node subscribes to /perception/person_detections and /perception/metrics,
maintains time-windowed state, and exposes ROS 2 services for querying:
- GetPeopleSnapshot: Current people in view
- GetSceneSummary: Scene statistics and health
- SelectFollowTarget: Target selection by strategy
"""

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time as RclpyTime
from builtin_interfaces.msg import Time as TimeMsg
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D

from thor_msgs.msg import PerceptionMetrics, PersonState
from thor_msgs.srv import GetPeopleSnapshot, GetSceneSummary, SelectFollowTarget
from thor_behavior import (
    WatchdogThread,
    DegradationPolicy, DegradedState, DepPolicy, FailureMode, RecoveryTarget,
    ResponseType,
)
from thor_telemetry import get_logger as _get_structured_logger, ErrorCode
from thor_telemetry import BoundaryContract, BoundaryType, ErrorBehavior
from thor_perception.health.ros_health_publisher import RosHealthPublisher

_structured_logger = _get_structured_logger("perception_state_server")

_deg_policy = DegradationPolicy("perception_state_server", [
    DepPolicy(
        dependency="perception_topics",
        failure_modes=[FailureMode.DATA_STALE, FailureMode.TIMEOUT],
        response=ResponseType.DEGRADE_WITH_NOTIFICATION,
        recovery_target=RecoveryTarget.FULL_OPERATION,
        recovery_action="Wait for detection pipeline to publish on /perception/person_detections",
    ),
])
_deg_state = DegradedState(_deg_policy, logger=_structured_logger)

# --- CP-010 Boundary Contracts (SI-11.1) ---
GET_SNAPSHOT_CONTRACT = BoundaryContract(
    boundary_name="perception_get_snapshot",
    boundary_type=BoundaryType.IPC,
    timeout_sec=1.0,
    error_behavior=ErrorBehavior.FAIL,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.INTERNAL}),
)

GET_SUMMARY_CONTRACT = BoundaryContract(
    boundary_name="perception_get_summary",
    boundary_type=BoundaryType.IPC,
    timeout_sec=1.0,
    error_behavior=ErrorBehavior.FAIL,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.INTERNAL}),
)

ON_DETECTIONS_CONTRACT = BoundaryContract(
    boundary_name="perception_on_detections",
    boundary_type=BoundaryType.IPC,
    timeout_sec=0.05,
    error_behavior=ErrorBehavior.DEGRADE,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.INTERNAL}),
)


class DetectionSample(NamedTuple):
    """Single detection sample for rolling window."""
    timestamp: float
    center_x: float
    center_y: float
    confidence: float
    bbox: BoundingBox2D


@dataclass
class TrackState:
    """Internal state for a tracked person."""
    track_id: int
    first_seen: float
    last_seen: float
    detection_count: int = 0
    samples: deque = field(default_factory=lambda: deque(maxlen=500))

    @property
    def latest_bbox(self) -> Optional[BoundingBox2D]:
        return self.samples[-1].bbox if self.samples else None

    @property
    def latest_confidence(self) -> float:
        return self.samples[-1].confidence if self.samples else 0.0

    @property
    def bbox_area(self) -> float:
        bbox = self.latest_bbox
        return bbox.size_x * bbox.size_y if bbox else 0.0

    def median_movement(self) -> float:
        """Compute median frame-to-frame movement in pixels."""
        if len(self.samples) < 2:
            return 0.0
        movements = []
        samples_list = list(self.samples)
        for i in range(1, len(samples_list)):
            dx = samples_list[i].center_x - samples_list[i - 1].center_x
            dy = samples_list[i].center_y - samples_list[i - 1].center_y
            movements.append((dx * dx + dy * dy) ** 0.5)
        movements.sort()
        return movements[len(movements) // 2]

    def median_movement_per_sec(self) -> float:
        """Median movement in pixels/second (FPS-normalized)."""
        if len(self.samples) < 2:
            return 0.0
        movements_per_sec = []
        samples_list = list(self.samples)
        for i in range(1, len(samples_list)):
            dx = samples_list[i].center_x - samples_list[i - 1].center_x
            dy = samples_list[i].center_y - samples_list[i - 1].center_y
            dt = samples_list[i].timestamp - samples_list[i - 1].timestamp
            if dt > 0:
                speed = ((dx * dx + dy * dy) ** 0.5) / dt
                movements_per_sec.append(speed)
        if not movements_per_sec:
            return 0.0
        movements_per_sec.sort()
        return movements_per_sec[len(movements_per_sec) // 2]


class PerceptionStateServer(Node, RosHealthPublisher):
    """Aggregates detections and provides query services."""

    def __init__(self):
        super().__init__('perception_state_server')
        self._health_init('perception_state_server')

        # Parameters
        self.declare_parameter('retention_sec', 30.0)
        self.declare_parameter('cleanup_interval_sec', 10.0)
        self.declare_parameter('stationary_threshold_px', 20.0)
        self.declare_parameter('iou_association_thresh', 0.3)
        self.declare_parameter('association_max_gap_sec', 2.0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)

        # Camera intrinsics (for bearing computation)
        self.declare_parameter('camera_fx_pixels', 0.0)
        self.declare_parameter('camera_cx_pixels', 0.0)
        self.declare_parameter('camera_hfov_deg', 65.0)

        # Cache parameter values
        self.retention_sec = self.get_parameter('retention_sec').value
        self.stationary_threshold_px = self.get_parameter('stationary_threshold_px').value
        self.iou_association_thresh = self.get_parameter('iou_association_thresh').value
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value

        # Camera intrinsics for bearing computation
        camera_fx_pixels = self.get_parameter('camera_fx_pixels').value
        camera_cx_pixels = self.get_parameter('camera_cx_pixels').value
        camera_hfov_deg = self.get_parameter('camera_hfov_deg').value

        # Principal point (default to center)
        self.cx_pixels = camera_cx_pixels if camera_cx_pixels > 0 else (self.frame_width / 2)

        # Focal length with HFOV validation
        if camera_fx_pixels > 0:
            self.fx_pixels = camera_fx_pixels
        else:
            hfov = camera_hfov_deg
            if hfov <= 1.0 or hfov >= 179.0:
                self.get_logger().error(f"Invalid camera_hfov_deg={hfov}, using 65°")
                hfov = 65.0
            self.fx_pixels = (self.frame_width / 2) / math.tan(math.radians(hfov / 2))

        self.get_logger().info(
            f"Camera: fx={self.fx_pixels:.1f}px, cx={self.cx_pixels:.1f}px (hfov={camera_hfov_deg}°)"
        )

        # State storage (in-memory, protected by lock)
        self._tracks: dict[int, TrackState] = {}
        self._next_track_id: int = 1
        self._frame_times: deque[float] = deque(maxlen=1000)
        self._latest_metrics: Optional[PerceptionMetrics] = None
        self._lock = threading.Lock()

        # QoS profile matching deepstream publisher (BEST_EFFORT for high-frequency sensor data)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers (default callback group)
        self.det_sub = self.create_subscription(
            Detection2DArray, '/perception/person_detections',
            self._on_detections, sensor_qos
        )
        self.metrics_sub = self.create_subscription(
            PerceptionMetrics, '/perception/metrics',
            self._on_metrics, 10
        )

        # Services (ReentrantCallbackGroup for concurrent service calls)
        self.service_group = ReentrantCallbackGroup()
        self.create_service(
            GetPeopleSnapshot, '/perception/get_snapshot',
            self._handle_get_snapshot, callback_group=self.service_group
        )
        self.create_service(
            GetSceneSummary, '/perception/get_summary',
            self._handle_get_summary, callback_group=self.service_group
        )
        self.create_service(
            SelectFollowTarget, '/perception/select_target',
            self._handle_select_target, callback_group=self.service_group
        )

        # Cleanup timer
        self.cleanup_timer = self.create_timer(
            self.get_parameter('cleanup_interval_sec').value,
            self._cleanup_old_tracks
        )

        # Watchdog: independent liveness check (CP-002)
        self._watchdog = WatchdogThread(
            name="perception_state_server",
            interval_sec=5.0,
            on_timeout=self._on_watchdog_timeout,
        )
        self._stale = False
        self._watchdog.start()

        self.get_logger().info(
            f'PerceptionStateServer started (retention={self.retention_sec}s, '
            f'frame={self.frame_width}x{self.frame_height})'
        )

    def _msg_time_to_float(self, stamp: TimeMsg) -> float:
        """Convert ROS Time message to float seconds."""
        return stamp.sec + stamp.nanosec * 1e-9

    def _float_to_msg_time(self, ts: float) -> TimeMsg:
        """Convert float seconds to ROS Time message."""
        msg = TimeMsg()
        msg.sec = int(ts)
        msg.nanosec = int((ts - int(ts)) * 1e9)
        return msg

    def _parse_track_id(self, det_id) -> Optional[int]:
        """Parse Detection2D.id as numeric track ID, or None if unset.

        Type-agnostic: accepts int, str, or unset.
        Treats 0 or empty string as "no tracker id".
        Only warns if non-empty string can't parse.
        """
        if isinstance(det_id, int):
            return det_id if det_id != 0 else None

        if isinstance(det_id, str):
            if not det_id or det_id == "":
                return None
            try:
                track_id = int(det_id)
                return track_id if track_id != 0 else None
            except ValueError:
                self.get_logger().warn(
                    f"Non-numeric Detection2D.id: {det_id!r}, falling back to IoU"
                )
                return None

        return None

    def _on_watchdog_timeout(self, name: str, elapsed_sec: float) -> None:
        """Watchdog timeout: mark data as stale."""
        self.get_logger().warning(
            f"Watchdog timeout: no detections for {elapsed_sec:.1f}s — data stale"
        )
        _structured_logger.emit_failure(
                trace_ctx=None,  # CP-006 §5.1: perception has no turn-level trace
                operation="detection_pipeline",
            error_code=ErrorCode.TIMEOUT,
            error_detail=f"No detections for {elapsed_sec:.1f}s",
            trigger="watchdog_timeout",
            elapsed_sec=round(elapsed_sec, 1),
        )
        self._stale = True
        _response = _deg_state.report_degradation(
            "perception_topics", FailureMode.DATA_STALE,
            f"No detections for {elapsed_sec:.1f}s",
        )

    def _health_ok(self) -> bool:
        """Healthy if watchdog hasn't flagged stale."""
        return not getattr(self, '_stale', False)

    def _health_status(self) -> str:
        """CP-009: report 'degraded' when dependency is degraded."""
        if _deg_state.is_degraded():
            return "degraded"
        return "healthy" if self._health_ok() else "unhealthy"

    def _on_detections(self, msg: Detection2DArray):
        """Ingest detections, associating with existing tracks."""
        t0 = time.monotonic()
        self._watchdog.heartbeat()
        self._stale = False
        _deg_state.report_recovery("perception_topics")
        with self._lock:
            timestamp = self._msg_time_to_float(msg.header.stamp)

            # Record frame timestamp for presence_ratio calculation
            self._frame_times.append(timestamp)

            matched_this_frame: set[int] = set()

            for det in msg.detections:
                track_id = self._parse_track_id(det.id)
                if track_id is None:
                    track_id = self._associate_by_iou(det, matched_this_frame, timestamp)

                matched_this_frame.add(track_id)
                self._update_track(track_id, det, timestamp)
        timing_ms = round((time.monotonic() - t0) * 1000, 1)
        if timing_ms > ON_DETECTIONS_CONTRACT.timeout_sec * 1000:
            _structured_logger.warning(
                "boundary_budget_overrun",
                trace_ctx=None,
                **ON_DETECTIONS_CONTRACT.log_fields(),
                timing_ms=timing_ms,
            )

    def _on_metrics(self, msg: PerceptionMetrics):
        """Cache latest metrics for scene summary."""
        with self._lock:
            self._latest_metrics = msg

    def _associate_by_iou(
        self, det: Detection2D, already_matched: set[int], now_ts: float
    ) -> int:
        """Simple IoU-based association when tracker is off."""
        best_iou = 0.0
        best_track_id = None
        max_gap = self.get_parameter('association_max_gap_sec').value

        for tid, track in self._tracks.items():
            if tid in already_matched:
                continue
            if now_ts - track.last_seen > max_gap:
                continue
            iou = self._compute_iou(det.bbox, track.latest_bbox)
            if iou > best_iou and iou > self.iou_association_thresh:
                best_iou = iou
                best_track_id = tid

        if best_track_id is not None:
            return best_track_id

        new_id = self._next_track_id
        self._next_track_id += 1
        return new_id

    def _update_track(self, track_id: int, det: Detection2D, timestamp: float):
        """Update or create track state with new detection."""
        bbox = det.bbox
        center_x = bbox.center.position.x
        center_y = bbox.center.position.y
        confidence = det.results[0].hypothesis.score if det.results else 0.0

        sample = DetectionSample(
            timestamp=timestamp,
            center_x=center_x,
            center_y=center_y,
            confidence=confidence,
            bbox=bbox
        )

        if track_id not in self._tracks:
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                first_seen=timestamp,
                last_seen=timestamp,
                detection_count=1,
            )
            self._tracks[track_id].samples.append(sample)
        else:
            track = self._tracks[track_id]
            track.last_seen = timestamp
            track.detection_count += 1
            track.samples.append(sample)

    def _compute_iou(
        self, bbox1: Optional[BoundingBox2D], bbox2: Optional[BoundingBox2D]
    ) -> float:
        """Compute intersection-over-union of two bounding boxes."""
        if bbox1 is None or bbox2 is None:
            return 0.0

        x1_min = bbox1.center.position.x - bbox1.size_x / 2
        x1_max = bbox1.center.position.x + bbox1.size_x / 2
        y1_min = bbox1.center.position.y - bbox1.size_y / 2
        y1_max = bbox1.center.position.y + bbox1.size_y / 2

        x2_min = bbox2.center.position.x - bbox2.size_x / 2
        x2_max = bbox2.center.position.x + bbox2.size_x / 2
        y2_min = bbox2.center.position.y - bbox2.size_y / 2
        y2_max = bbox2.center.position.y + bbox2.size_y / 2

        inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter_area = inter_x * inter_y

        area1 = bbox1.size_x * bbox1.size_y
        area2 = bbox2.size_x * bbox2.size_y
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _distance_to_center(self, track: TrackState) -> float:
        """Euclidean distance from track's latest center to frame center."""
        bbox = track.latest_bbox
        if bbox is None:
            return float('inf')
        frame_cx = self.frame_width / 2
        frame_cy = self.frame_height / 2
        dx = bbox.center.position.x - frame_cx
        dy = bbox.center.position.y - frame_cy
        return (dx * dx + dy * dy) ** 0.5

    def _compute_stability_score(self, track: TrackState, cutoff: float) -> float:
        """
        Composite quality metric (0-1):
        - age_factor: longer tracking = more stable (saturates at 5s)
        - continuity_factor: presence_ratio in window
        - smoothness_factor: inverse of median bbox movement (normalized by dt)
        """
        age_sec = track.last_seen - track.first_seen
        age_factor = min(1.0, age_sec / 5.0)  # Saturate at 5s

        samples_in_window = [s for s in track.samples if s.timestamp >= cutoff]
        frames_in_window = sum(1 for t in self._frame_times if t >= cutoff)
        continuity_factor = len(samples_in_window) / max(1, frames_in_window)

        # Smoothness: normalize movement by dt (pixels/second, not pixels/frame)
        # This prevents FPS drops from making tracks look falsely smooth
        movement_per_sec = track.median_movement_per_sec()
        smoothness_factor = 1.0 / (1.0 + movement_per_sec / 100.0)  # 100px/s → 0.5

        # Weighted combination
        return 0.3 * age_factor + 0.4 * continuity_factor + 0.3 * smoothness_factor

    def _compute_primary_candidate_score(
        self, track: TrackState, cutoff: float, latest_frame_ts: float
    ) -> float:
        """
        Policy score for follow target selection.
        Blends stability with spatial preference (center/size).
        Penalizes stale tracks.
        """
        stability = self._compute_stability_score(track, cutoff)

        # Staleness penalty: if last_seen_ms > 500ms, penalize heavily
        last_seen_ms = max(0, int((latest_frame_ts - track.last_seen) * 1000))
        if last_seen_ms < 500:
            staleness_penalty = 1.0
        elif last_seen_ms < 1000:
            staleness_penalty = 0.5
        else:
            staleness_penalty = 0.1

        # Center score: prefer tracks near frame center
        dist = self._distance_to_center(track)
        center_score = 1.0 / (1.0 + dist / 200.0)  # 200px → 0.5

        # Combine: 60% stability, 25% center preference, apply staleness penalty
        return (0.6 * stability + 0.25 * center_score) * staleness_penalty

    def _compute_bearing_deg(self, center_x: float) -> float:
        """Horizontal bearing from optical axis. Positive = right."""
        if self.fx_pixels < 1e-3:
            return 0.0  # Invalid fx, return 0
        pixel_offset = center_x - self.cx_pixels
        return math.degrees(math.atan2(pixel_offset, self.fx_pixels))

    def _compute_bearing_std_deg(self, track: TrackState, cutoff: float) -> float:
        """
        Bearing uncertainty: std of residuals after removing linear trend.
        This isolates detector noise from actual person motion.
        """
        if self.fx_pixels < 1e-3:
            return 30.0  # Invalid fx, return conservative default

        samples = [s for s in track.samples if s.timestamp >= cutoff]
        if len(samples) < 2:
            return 30.0  # Conservative default when insufficient data

        # Compute bearings and timestamps
        bearings = [self._compute_bearing_deg(s.center_x) for s in samples]
        times = [s.timestamp for s in samples]
        t0 = times[0]
        times = [t - t0 for t in times]  # Normalize to start at 0

        # Fit linear trend: bearing = slope * t + intercept
        n = len(samples)
        sum_t = sum(times)
        sum_b = sum(bearings)
        sum_tb = sum(t * b for t, b in zip(times, bearings))
        sum_t2 = sum(t * t for t in times)

        denom = n * sum_t2 - sum_t * sum_t
        if abs(denom) < 1e-9:
            # All samples at same time, use raw variance (n-1 for unbiased)
            mean_b = sum_b / n
            variance = sum((b - mean_b) ** 2 for b in bearings) / max(1, n - 1)
            return min(30.0, variance ** 0.5)

        slope = (n * sum_tb - sum_t * sum_b) / denom
        intercept = (sum_b - slope * sum_t) / n

        # Compute residuals (bearing - trend), use n-2 for linear fit (2 params)
        residuals = [b - (slope * t + intercept) for t, b in zip(times, bearings)]
        variance = sum(r * r for r in residuals) / max(1, n - 2)
        return min(30.0, variance ** 0.5)  # Cap at 30°

    def _track_to_person_state(
        self, track: TrackState, cutoff: float, latest_frame_ts: float
    ) -> PersonState:
        """Convert internal TrackState to PersonState message."""
        msg = PersonState()
        msg.track_id = track.track_id
        msg.first_seen = self._float_to_msg_time(track.first_seen)
        msg.last_seen = self._float_to_msg_time(track.last_seen)
        msg.detection_count = track.detection_count
        msg.bbox = track.latest_bbox or BoundingBox2D()
        msg.confidence = track.latest_confidence

        samples_in_window = [s for s in track.samples if s.timestamp >= cutoff]
        frames_in_window = sum(1 for t in self._frame_times if t >= cutoff)
        msg.presence_ratio = len(samples_in_window) / max(1, frames_in_window)
        msg.time_visible_sec = track.last_seen - max(track.first_seen, cutoff)
        msg.is_stationary = track.median_movement() < self.stationary_threshold_px

        # Quality-aware tracking fields
        # IMPORTANT: Use latest_frame_ts (from _frame_times), not wall time!
        msg.age_ms = max(0, int((latest_frame_ts - track.first_seen) * 1000))
        msg.last_seen_ms = max(0, int((latest_frame_ts - track.last_seen) * 1000))
        msg.stability_score = self._compute_stability_score(track, cutoff)
        # is_primary_candidate set by caller after computing all candidates
        msg.is_primary_candidate = False

        # Spatial awareness fields
        msg.bearing_deg = self._compute_bearing_deg(track.latest_bbox.center.position.x)
        msg.bearing_std_deg = self._compute_bearing_std_deg(track, cutoff)
        msg.range_m = float('nan')      # No depth sensor
        msg.range_std_m = float('nan')  # No depth sensor
        msg.range_quality = 0.0         # 0 = range unavailable

        return msg

    def _handle_get_snapshot(self, request, response):
        """Handle GetPeopleSnapshot service request."""
        t0 = time.monotonic()
        with self._lock:
            # Guard: no frames yet - include health info for diagnostics
            if not self._frame_times:
                response.success = False
                # Include health context in error for diagnostics
                if self._latest_metrics:
                    health_map = {0: "OK", 1: "DEGRADED", 2: "ERROR"}
                    health_str = health_map.get(self._latest_metrics.health, "UNKNOWN")
                    reason = self._latest_metrics.reason_code or "UNKNOWN"
                    response.error = f"NO_FRAMES_RECEIVED (health={health_str}, reason={reason})"
                else:
                    response.error = "NO_FRAMES_RECEIVED"
                response.people = []
                response.total_frames_in_window = 0
                response.window_used_sec = 0.0
                return response

            latest_ts = max(self._frame_times)

            # Use requested window, not always retention_sec
            if request.latest_only:
                cutoff = latest_ts - 0.001  # 1ms epsilon
                window_used = 0.001
            else:
                window_used = min(request.max_age_sec, self.retention_sec)
                cutoff = latest_ts - window_used

            candidates = [t for t in self._tracks.values() if t.last_seen >= cutoff]
            frames_in_window = sum(1 for t in self._frame_times if t >= cutoff)

            response.success = True
            response.error = ""
            # CP-003 (SI-5.3): timestamp reflects newest sample time, not current wall clock.
            # Clients compute data age as (now - response.timestamp).
            response.timestamp = self._float_to_msg_time(latest_ts)
            response.people = [
                self._track_to_person_state(t, cutoff, latest_ts) for t in candidates
            ]
            response.total_frames_in_window = frames_in_window
            response.window_used_sec = window_used

            # Mark primary by track_id (not by list position!)
            if response.people:
                # Find primary track_id using policy score
                primary_id = max(
                    candidates,
                    key=lambda t: self._compute_primary_candidate_score(t, cutoff, latest_ts)
                ).track_id
                for ps in response.people:
                    ps.is_primary_candidate = (ps.track_id == primary_id)

        timing_ms = round((time.monotonic() - t0) * 1000, 1)
        if timing_ms > GET_SNAPSHOT_CONTRACT.timeout_sec * 1000:
            _structured_logger.warning(
                "boundary_budget_overrun",
                trace_ctx=None,
                **GET_SNAPSHOT_CONTRACT.log_fields(),
                timing_ms=timing_ms,
            )
        return response

    def _handle_get_summary(self, request, response):
        """Handle GetSceneSummary service request."""
        t0 = time.monotonic()
        with self._lock:
            # Guard: no frames yet
            if not self._frame_times:
                response.success = False
                response.error = "NO_FRAMES_RECEIVED"
                response.people_count = 0
                response.total_detections = 0
                response.avg_confidence = 0.0
                response.scene_stability = 0.0
                response.primary_track_id = 0
                response.primary_quality = 0.0
                response.primary_stability_score = 0.0
                response.primary_last_seen_ms = 0
                response.track_ids = []
                response.fps = 0.0
                response.latency_p95_ms = 0.0
                response.health = 0
                response.reason_code = ""
                response.text_summary = "error=NO_FRAMES_RECEIVED"
                response.window_used_sec = 0.0
                response.timestamp = self._float_to_msg_time(time.time())
                return response

            latest_ts = max(self._frame_times)
            window_used = min(request.window_sec, self.retention_sec)
            cutoff = latest_ts - window_used

            candidates = [t for t in self._tracks.values() if t.last_seen >= cutoff]

            response.success = True
            response.error = ""
            # CP-003 (SI-5.3): timestamp reflects newest sample time
            response.timestamp = self._float_to_msg_time(latest_ts)
            response.window_used_sec = window_used

            response.people_count = len(candidates)
            response.track_ids = [t.track_id for t in candidates]

            # Compute total detections and avg confidence
            total_detections = 0
            total_confidence = 0.0
            for t in candidates:
                samples_in_window = [s for s in t.samples if s.timestamp >= cutoff]
                total_detections += len(samples_in_window)
                total_confidence += sum(s.confidence for s in samples_in_window)

            response.total_detections = total_detections
            response.avg_confidence = (
                total_confidence / total_detections if total_detections > 0 else 0.0
            )

            # Compute scene stability
            if candidates:
                avg_movement = sum(t.median_movement() for t in candidates) / len(candidates)
                stability = 1.0 - min(1.0, avg_movement / 100.0)
            else:
                stability = 1.0
            response.scene_stability = stability

            # Primary target
            if candidates:
                def quality_score(t):
                    conf = t.latest_confidence
                    samples_in_window = [s for s in t.samples if s.timestamp >= cutoff]
                    frames_in_window = sum(1 for ts in self._frame_times if ts >= cutoff)
                    presence = len(samples_in_window) / max(1, frames_in_window)
                    stationary_bonus = 1.0 if t.median_movement() < self.stationary_threshold_px else 0.7
                    return conf * presence * stationary_bonus

                primary = max(candidates, key=quality_score)
                response.primary_track_id = primary.track_id
                response.primary_quality = quality_score(primary)
                response.primary_stability_score = self._compute_stability_score(primary, cutoff)
                response.primary_last_seen_ms = max(0, int((latest_ts - primary.last_seen) * 1000))
            else:
                response.primary_track_id = 0
                response.primary_quality = 0.0
                response.primary_stability_score = 0.0
                response.primary_last_seen_ms = 0

            # Pipeline health from metrics
            if self._latest_metrics:
                response.fps = self._latest_metrics.fps
                response.latency_p95_ms = self._latest_metrics.latency_p95_ms
                response.health = self._latest_metrics.health
                response.reason_code = self._latest_metrics.reason_code
            else:
                response.fps = 0.0
                response.latency_p95_ms = 0.0
                response.health = 0
                response.reason_code = ""

            # Text summary
            stability_label = "HIGH" if stability > 0.7 else ("MED" if stability > 0.4 else "LOW")
            health_label = ["OK", "DEGRADED", "ERROR"][response.health] if response.health < 3 else "UNKNOWN"
            response.text_summary = (
                f"people={response.people_count}; "
                f"primary=track_{response.primary_track_id}; "
                f"stability={stability_label}; "
                f"fps={response.fps:.1f}; "
                f"latency_p95={response.latency_p95_ms:.0f}ms; "
                f"health={health_label}"
            )

        timing_ms = round((time.monotonic() - t0) * 1000, 1)
        if timing_ms > GET_SUMMARY_CONTRACT.timeout_sec * 1000:
            _structured_logger.warning(
                "boundary_budget_overrun",
                trace_ctx=None,
                **GET_SUMMARY_CONTRACT.log_fields(),
                timing_ms=timing_ms,
            )
        return response

    def _handle_select_target(self, request, response):
        """Handle SelectFollowTarget service request."""
        t0 = time.monotonic()
        with self._lock:
            latest_ts = max(self._frame_times) if self._frame_times else 0.0
            cutoff = latest_ts - self.retention_sec
            candidates = [t for t in self._tracks.values() if t.last_seen >= cutoff]

            if not candidates:
                response.success = False
                response.error = "NO_PEOPLE_DETECTED"
                response.selected_track_id = 0
                response.target = PersonState()
                return response

            selected = None
            if request.strategy == "nearest_center":
                selected = min(candidates, key=lambda t: self._distance_to_center(t))
            elif request.strategy == "most_confident":
                selected = max(candidates, key=lambda t: t.latest_confidence)
            elif request.strategy == "largest":
                selected = max(candidates, key=lambda t: t.bbox_area)
            elif request.strategy == "by_track_id":
                selected = self._tracks.get(request.track_id)
                if selected is None or selected.last_seen < cutoff:
                    response.success = False
                    response.error = "TRACK_NOT_FOUND"
                    response.selected_track_id = 0
                    response.target = PersonState()
                    return response
            else:
                response.success = False
                response.error = f"UNKNOWN_STRATEGY: {request.strategy}"
                response.selected_track_id = 0
                response.target = PersonState()
                return response

            response.success = True
            response.error = ""
            response.selected_track_id = selected.track_id
            response.target = self._track_to_person_state(selected, cutoff, latest_ts)

        timing_ms = round((time.monotonic() - t0) * 1000, 1)
        if timing_ms > GET_SNAPSHOT_CONTRACT.timeout_sec * 1000:
            _structured_logger.warning(
                "boundary_budget_overrun",
                trace_ctx=None,
                **GET_SNAPSHOT_CONTRACT.log_fields(),
                timing_ms=timing_ms,
            )
        return response

    def _cleanup_old_tracks(self):
        """Periodic cleanup: prune old samples and remove dead tracks."""
        with self._lock:
            cutoff = time.time() - self.retention_sec

            # Prune old frame times
            while self._frame_times and self._frame_times[0] < cutoff:
                self._frame_times.popleft()

            # Prune per-track samples and remove dead tracks
            dead_tracks = []
            for tid, track in self._tracks.items():
                while track.samples and track.samples[0].timestamp < cutoff:
                    track.samples.popleft()

                if not track.samples or track.last_seen < cutoff:
                    dead_tracks.append(tid)

            for tid in dead_tracks:
                del self._tracks[tid]

            if dead_tracks:
                self.get_logger().debug(f'Cleaned up {len(dead_tracks)} old tracks')


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionStateServer()

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


if __name__ == '__main__':
    main()
