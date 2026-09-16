#!/usr/bin/env python3
"""Person Tracker Node - Track-by-detection for person detections.

Subscribes to person detections and publishes stable tracked persons
with lifecycle state management and ambiguity signals.

Pipeline:
    /perception/person_detections -> [IoU Tracker] -> /tracking/person_tracks

Key contracts:
    - track_id is stable and monotonically increasing
    - Timestamp from detection header (NOT get_clock().now())
    - State transitions: TENTATIVE -> CONFIRMED -> OCCLUDED -> LOST
    - LOST tracks are NOT published
    - match_margin exposes ambiguity for Phase 3.3

Usage:
    ros2 run thor_perception person_tracker
    ros2 launch thor_perception identity_pipeline.launch.py
"""

import os
import sys
import time
import math
from collections import deque
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from vision_msgs.msg import Detection2DArray

from thor_msgs.msg import PersonTrack, PersonTracks, PerceptionMetrics
from thor_perception.tracking.person_tracker import (
    PersonTracker,
    PersonDetection,
    TrackedPerson,
    TrackState,
    TimeStatus,
    QualityReason,
)


class PersonTrackerNode(Node):
    """ROS 2 node for person tracking."""

    def __init__(self):
        super().__init__("person_tracker_node")

        # Declare parameters
        self.declare_parameter("camera_hfov_deg", 65.0)
        self.declare_parameter("publish_metrics", True)
        self.declare_parameter("metrics_interval_sec", 5.0)
        self.declare_parameter("min_fps_threshold", float(os.environ.get("PERCEPTION_MIN_FPS", "5.0")))

        # Tracker parameters
        self.declare_parameter("tracker.iou_threshold", 0.3)
        self.declare_parameter("tracker.dist_threshold", 0.20)
        self.declare_parameter("tracker.size_ratio_threshold", 0.5)
        self.declare_parameter("tracker.max_gap_tentative_sec", 0.5)
        self.declare_parameter("tracker.max_gap_confirmed_sec", 1.0)
        self.declare_parameter("tracker.min_hits_to_confirm", 3)
        self.declare_parameter("tracker.cost_weight_iou", 0.6)
        self.declare_parameter("tracker.cost_weight_dist", 0.4)
        self.declare_parameter("tracker.continuity_bias", 0.9)
        self.declare_parameter("tracker.margin_threshold", 0.1)
        self.declare_parameter("tracker.reassoc_cooldown_ms", 500)
        self.declare_parameter("tracker.vmax_normalized", 0.6)

        # Get parameters
        self.camera_hfov_deg = self.get_parameter("camera_hfov_deg").value
        self.publish_metrics = self.get_parameter("publish_metrics").value
        self.metrics_interval_sec = self.get_parameter("metrics_interval_sec").value
        self.min_fps_threshold = self.get_parameter("min_fps_threshold").value

        # Initialize tracker
        self.tracker = PersonTracker(
            iou_threshold=self.get_parameter("tracker.iou_threshold").value,
            dist_threshold=self.get_parameter("tracker.dist_threshold").value,
            size_ratio_threshold=self.get_parameter("tracker.size_ratio_threshold").value,
            max_gap_tentative_sec=self.get_parameter("tracker.max_gap_tentative_sec").value,
            max_gap_confirmed_sec=self.get_parameter("tracker.max_gap_confirmed_sec").value,
            min_hits_to_confirm=self.get_parameter("tracker.min_hits_to_confirm").value,
            cost_weight_iou=self.get_parameter("tracker.cost_weight_iou").value,
            cost_weight_dist=self.get_parameter("tracker.cost_weight_dist").value,
            continuity_bias=self.get_parameter("tracker.continuity_bias").value,
            margin_threshold=self.get_parameter("tracker.margin_threshold").value,
            reassoc_cooldown_sec=self.get_parameter("tracker.reassoc_cooldown_ms").value / 1000.0,
            vmax_normalized=self.get_parameter("tracker.vmax_normalized").value,
        )

        # QoS: Match person_detector output
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriber
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            "/perception/person_detections",
            self._detection_callback,
            sensor_qos,
        )

        # Publishers
        self.tracks_pub = self.create_publisher(
            PersonTracks,
            "/tracking/person_tracks",
            sensor_qos,
        )

        if self.publish_metrics:
            self.metrics_pub = self.create_publisher(
                PerceptionMetrics,
                "/tracking/person_tracker/metrics",
                10,
            )
            self.metrics_timer = self.create_timer(
                self.metrics_interval_sec,
                self._publish_metrics,
            )

        # Metrics tracking
        self.frame_count = 0
        self.frame_times: deque = deque(maxlen=100)
        self.process_latencies: deque = deque(maxlen=100)
        self.frame_width = 640
        self.frame_height = 480
        self.frame_size_initialized = False

        self.get_logger().info("Person tracker node initialized")

    def _detection_callback(self, msg: Detection2DArray):
        """Process incoming detections and publish tracks."""
        start_time = time.perf_counter()

        # Extract timestamp as float seconds
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Update frame size from detections if available
        if not self.frame_size_initialized and msg.detections:
            # Try to infer from first detection or use defaults
            # Better approach: get from CameraInfo topic
            self.tracker.set_frame_size(self.frame_width, self.frame_height)
            self.frame_size_initialized = True

        # Convert Detection2D to PersonDetection
        detections = []
        for idx, det in enumerate(msg.detections):
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            confidence = det.results[0].hypothesis.score if det.results else 0.5

            detections.append(PersonDetection(
                cx=cx, cy=cy, w=w, h=h,
                confidence=confidence,
                det_idx=idx,
            ))

        # Run tracker
        tracked = self.tracker.update(detections, stamp_sec)

        # Build output message
        out_msg = PersonTracks()
        out_msg.header = msg.header  # Preserve original timestamp

        for track in tracked:
            track_msg = self._track_to_msg(track, stamp_sec)
            out_msg.tracks.append(track_msg)

        # Pipeline metrics
        out_msg.fps = self._calculate_fps()
        out_msg.total_persons_tracked = self.tracker.total_tracks_created
        out_msg.active_track_count = len(tracked)
        out_msg.time_status = self.tracker.time_status.value

        self.tracks_pub.publish(out_msg)

        # Update timing metrics
        process_time = time.perf_counter() - start_time
        self.frame_count += 1
        self.frame_times.append(time.perf_counter())
        self.process_latencies.append(process_time * 1000)

        # Log periodically
        if self.frame_count % 100 == 1:
            self.get_logger().info(
                f"frame={self.frame_count} fps={out_msg.fps:.1f} "
                f"tracks={len(tracked)} process_ms={process_time*1000:.1f}"
            )

    def _track_to_msg(self, track: TrackedPerson, now_sec: float) -> PersonTrack:
        """Convert TrackedPerson to PersonTrack message."""
        msg = PersonTrack()
        msg.track_id = track.track_id

        # Timestamp from track
        msg.stamp.sec = int(track.last_seen)
        msg.stamp.nanosec = int((track.last_seen % 1) * 1e9)
        msg.frame_id = "camera_head_optical_frame"

        # Observed bbox
        msg.bbox.center.position.x = track.cx
        msg.bbox.center.position.y = track.cy
        msg.bbox.center.theta = 0.0
        msg.bbox.size_x = track.w
        msg.bbox.size_y = track.h

        # Predicted bbox (small lookahead)
        dt_predict = 0.033  # ~1 frame at 30 FPS
        pred_cx = track.cx + track.vx * dt_predict * self.frame_width
        pred_cy = track.cy + track.vy * dt_predict * self.frame_height
        msg.predicted_bbox.center.position.x = pred_cx
        msg.predicted_bbox.center.position.y = pred_cy
        msg.predicted_bbox.center.theta = 0.0
        msg.predicted_bbox.size_x = track.w
        msg.predicted_bbox.size_y = track.h

        # Tracking state
        msg.last_update_age_ms = int((now_sec - track.last_seen) * 1000)
        msg.state = track.state.value
        msg.quality = track.quality
        msg.quality_reason = track.quality_reason

        # Ambiguity
        # Handle inf for message (use large value)
        if math.isinf(track.match_margin):
            msg.match_margin = 1000.0  # Large value = unambiguous
        else:
            msg.match_margin = track.match_margin

        # Details
        msg.detection_count = track.detection_count
        msg.detector_confidence = track.confidence
        msg.stability_score = track.stability_score
        msg.vx = track.vx
        msg.vy = track.vy

        # Source info
        msg.source_width = self.frame_width
        msg.source_height = self.frame_height

        # Bearing
        msg.bearing_deg = self._compute_bearing(track.cx)

        return msg

    def _compute_bearing(self, center_x: float) -> float:
        """Compute horizontal bearing from camera center."""
        hfov_rad = self.camera_hfov_deg * math.pi / 180.0
        fx_pixels = (self.frame_width / 2) / math.tan(hfov_rad / 2)
        cx_pixels = self.frame_width / 2
        bearing_rad = math.atan2(center_x - cx_pixels, fx_pixels)
        return bearing_rad * 180.0 / math.pi

    def _calculate_fps(self) -> float:
        """Calculate current FPS from frame times."""
        if len(self.frame_times) < 2:
            return 0.0
        times = list(self.frame_times)
        duration = times[-1] - times[0]
        return (len(times) - 1) / duration if duration > 0 else 0.0

    def _publish_metrics(self):
        """Publish perception metrics."""
        msg = PerceptionMetrics()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "person_tracker"

        msg.fps = self._calculate_fps()
        if self.process_latencies:
            msg.latency_p95_ms = float(np.percentile(list(self.process_latencies), 95))
        else:
            msg.latency_p95_ms = 0.0

        msg.model_name = "PersonTracker"
        msg.model_version = "v3.1"
        msg.tracker_enabled = True

        # Health: OK if receiving detections at reasonable rate
        if msg.fps < self.min_fps_threshold and self.frame_count > 10:
            msg.health = PerceptionMetrics.DEGRADED
            msg.reason_code = "LOW_INPUT_RATE"
        elif self.tracker.time_status != TimeStatus.OK:
            msg.health = PerceptionMetrics.DEGRADED
            msg.reason_code = "TIME_ANOMALY"
        else:
            msg.health = PerceptionMetrics.OK
            msg.reason_code = ""

        self.metrics_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
