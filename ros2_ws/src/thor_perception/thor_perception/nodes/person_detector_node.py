#!/usr/bin/env python3
"""Person Detection Node - TensorRT PeopleNet inference.

Subscribes to camera_ingest image topics and publishes person detections.
This is a stateless perception primitive: image in -> detections out.

Architecture:
    /sensors/camera/head/rgb/image -> [TensorRT inference] -> /perception/person_detections

Key contracts:
    - Detection header.stamp comes from input image (NOT get_clock().now())
    - Bbox coordinates are in original image pixel space
    - No tracking or target selection (that's Phase 6)

Usage:
    ros2 run thor_perception person_detector
    ros2 launch thor_perception identity_pipeline.launch.py
"""

import os
import sys
import time
from collections import deque
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
from geometry_msgs.msg import Pose2D

from thor_msgs.msg import PerceptionMetrics
from thor_perception.inference.peoplenet_trt import PeopleNetTRT, create_detector
from thor_telemetry import get_logger as _get_structured_logger, ErrorCode

_structured_logger = _get_structured_logger("person_detector_node")


# Engine paths
DEFAULT_ENGINE_PATH = "/opt/models/perception/resnet34_peoplenet.onnx_b1_gpu0_fp16.engine"
DEFAULT_ONNX_PATH = "/opt/models/perception/resnet34_peoplenet.onnx"


class PersonDetectorNode(Node):
    """ROS 2 node for TensorRT-accelerated person detection."""

    def __init__(self):
        super().__init__("person_detector_node")

        # Declare parameters
        self.declare_parameter("engine_path", DEFAULT_ENGINE_PATH)
        self.declare_parameter("onnx_path", DEFAULT_ONNX_PATH)
        self.declare_parameter("min_confidence", 0.5)
        self.declare_parameter("nms_iou_threshold", 0.4)
        self.declare_parameter("publish_metrics", True)
        self.declare_parameter("metrics_interval_sec", 5.0)
        self.declare_parameter("min_fps_threshold", float(os.environ.get("PERCEPTION_MIN_FPS", "5.0")))

        # Get parameters
        self.engine_path = self.get_parameter("engine_path").value
        self.onnx_path = self.get_parameter("onnx_path").value
        self.min_confidence = self.get_parameter("min_confidence").value
        self.nms_iou_threshold = self.get_parameter("nms_iou_threshold").value
        self.publish_metrics = self.get_parameter("publish_metrics").value
        self.metrics_interval_sec = self.get_parameter("metrics_interval_sec").value
        self.min_fps_threshold = self.get_parameter("min_fps_threshold").value

        # Initialize detector (fails clearly if engine missing)
        self.detector: Optional[PeopleNetTRT] = None
        self.engine_hash = ""

        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()

        # QoS for sensor data: best-effort, keep last 1 (latest-frame semantics)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe to camera image
        self.image_sub = self.create_subscription(
            Image,
            "/sensors/camera/head/rgb/image",
            self._image_callback,
            sensor_qos,
        )

        # Publisher for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            "/perception/person_detections",
            sensor_qos,
        )

        # Metrics publisher
        if self.publish_metrics:
            self.metrics_pub = self.create_publisher(
                PerceptionMetrics, "/perception/person_detector/metrics", 10
            )
            self.metrics_timer = self.create_timer(
                self.metrics_interval_sec, self._publish_metrics
            )

        # Metrics tracking
        self.frame_count = 0
        self.frame_times = deque(maxlen=100)
        self.compute_latencies = deque(maxlen=100)  # preprocess + infer + postprocess
        self.pipeline_ages = deque(maxlen=100)  # now - image.header.stamp

        # Health state
        self.health = PerceptionMetrics.OK
        self.reason_code = ""
        self.last_image_time: Optional[float] = None

        self.get_logger().info(
            f"Person detector initialized. Engine: {self.engine_path}"
        )

    def _init_detector(self) -> bool:
        """Initialize TensorRT detector. Called lazily on first image."""
        try:
            self.detector = create_detector(
                engine_path=self.engine_path,
                onnx_path=self.onnx_path,
                conf_threshold=self.min_confidence,
                nms_iou_threshold=self.nms_iou_threshold,
                logger=self.get_logger(),
            )
            self.engine_hash = self.detector.engine_hash
            self.health = PerceptionMetrics.OK
            self.reason_code = ""
            return True
        except RuntimeError as e:
            self.get_logger().error(f"Failed to initialize detector: {e}")
            _structured_logger.emit_failure(
                trace_ctx=None,  # CP-006 §5.1: perception has no turn-level trace
                operation="init_detector",
                error_code=ErrorCode.UNAVAILABLE,
                error_detail=str(e)[:200],
                trigger="RuntimeError",
            )
            self.health = PerceptionMetrics.ERROR
            self.reason_code = "ENGINE_MISSING"
            return False

    def _image_callback(self, msg: Image):
        """Process incoming image and publish detections."""
        callback_start = time.perf_counter()

        # Lazy initialization of detector
        if self.detector is None:
            if not self._init_detector():
                return

        # Convert ROS Image to OpenCV BGR
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"Failed to convert image: {e}")
            _structured_logger.emit_failure(
                trace_ctx=None,  # CP-006 §5.1: perception has no turn-level trace
                operation="image_conversion",
                error_code=ErrorCode.INTERNAL,
                error_detail=str(e)[:200],
                trigger=type(e).__name__,
            )
            return

        # Run detection with timing
        detections, preprocess_ms, infer_ms, postprocess_ms = self.detector.detect_timed(
            cv_image
        )

        # Build Detection2DArray message
        # CRITICAL: Use image timestamp, NOT get_clock().now()
        det_msg = Detection2DArray()
        det_msg.header.stamp = msg.header.stamp
        det_msg.header.frame_id = msg.header.frame_id

        for idx, det in enumerate(detections):
            d = Detection2D()
            d.id = str(idx)  # Stable per-detection index for Phase 3 association

            # Bbox in original image pixel space
            d.bbox.center.position.x = det.cx
            d.bbox.center.position.y = det.cy
            d.bbox.center.theta = 0.0
            d.bbox.size_x = det.width
            d.bbox.size_y = det.height

            # Hypothesis: person with confidence
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = "person"
            hyp.hypothesis.score = det.score
            d.results.append(hyp)

            det_msg.detections.append(d)

        # Publish detections
        self.detection_pub.publish(det_msg)

        # Update metrics
        callback_end = time.perf_counter()
        compute_ms = preprocess_ms + infer_ms + postprocess_ms

        # Pipeline age: how old is the image when we finish processing?
        now = self.get_clock().now()
        image_stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        now_sec = now.nanoseconds * 1e-9
        pipeline_age_ms = (now_sec - image_stamp_sec) * 1000

        self.frame_count += 1
        self.frame_times.append(callback_end)
        self.compute_latencies.append(compute_ms)
        self.pipeline_ages.append(pipeline_age_ms)
        self.last_image_time = callback_end

        # Log every 100 frames
        if self.frame_count % 100 == 1:
            fps = self._calculate_fps()
            self.get_logger().info(
                f"frame={self.frame_count} fps={fps:.1f} compute_ms={compute_ms:.1f} "
                f"age_ms={pipeline_age_ms:.1f} detections={len(detections)}"
            )

    def _calculate_fps(self) -> float:
        """Calculate current FPS from frame times."""
        if len(self.frame_times) < 2:
            return 0.0
        times = list(self.frame_times)
        duration = times[-1] - times[0]
        return (len(times) - 1) / duration if duration > 0 else 0.0

    def _publish_metrics(self):
        """Publish perception metrics."""
        if not self.publish_metrics:
            return

        msg = PerceptionMetrics()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "person_detector"

        msg.fps = self._calculate_fps()

        # Compute latency P95 (what this node does)
        if self.compute_latencies:
            msg.latency_p95_ms = float(np.percentile(list(self.compute_latencies), 95))
        else:
            msg.latency_p95_ms = 0.0

        # Pipeline age P95 (end-to-end including camera_ingest, DDS)
        if self.pipeline_ages:
            pipeline_age_p95 = float(np.percentile(list(self.pipeline_ages), 95))
            pipeline_age_p99 = float(np.percentile(list(self.pipeline_ages), 99))
        else:
            pipeline_age_p95 = 0.0
            pipeline_age_p99 = 0.0

        msg.model_name = "PeopleNet"
        msg.model_version = self.engine_hash

        # Health assessment
        if self.detector is None:
            msg.health = PerceptionMetrics.ERROR
            msg.reason_code = "ENGINE_MISSING"
        elif msg.fps < self.min_fps_threshold and self.frame_count > 10:
            msg.health = PerceptionMetrics.DEGRADED
            msg.reason_code = "LOW_FPS"
        elif pipeline_age_p99 > 150.0 and self.frame_count > 10:
            msg.health = PerceptionMetrics.DEGRADED
            msg.reason_code = "HIGH_LATENCY"
        else:
            msg.health = PerceptionMetrics.OK
            msg.reason_code = ""

        self.health = msg.health
        self.reason_code = msg.reason_code

        self.metrics_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()

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
