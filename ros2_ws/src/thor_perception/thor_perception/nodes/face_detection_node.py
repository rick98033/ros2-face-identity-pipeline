#!/usr/bin/env python3
"""Face Detection Node - YuNet face detection with TensorRT (default) or ONNX Runtime (fallback).

Pipeline: ROS Image → YuNet (TensorRT/ONNX) → ROS 2 FaceTracks

Subscribes:
  /sensors/camera/head/rgb/image (sensor_msgs/Image) - From camera_ingest
  /tracking/person_tracks (thor_msgs/PersonTracks) - For face-to-person association

Publishes:
  /perception/faces/tracks (thor_msgs/FaceTracks)
  /perception/faces/metrics (thor_msgs/PerceptionMetrics)

Phase 3.2: Migrated from direct RTSP decode to ROS image subscription.
Face-to-person association populates associated_person_track_id field.

INFERENCE MODES:
  1. TensorRT (default): Uses pre-built .engine file for GPU-accelerated inference.
     Build engine once with: /opt/thor_perception/scripts/build_face_engine.sh

  2. ONNX CPU (fallback): Only enabled if FACE_PIPELINE_ALLOW_CPU_FALLBACK=true
     AND TensorRT engine is missing. Emits loud warnings when active.

Environment Variables:
  FACE_PIPELINE_ALLOW_CPU_FALLBACK: Set to 'true' to allow CPU fallback (default: false)
"""

import os
import random
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol
from pathlib import Path
import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from thor_perception.tracking.iou_tracker import IoUTracker, TrackedFace, RawDetection
from thor_perception.tracking.crop_buffer import FaceCropBuffer
from thor_perception.tracking.face_person_association import (
    PersonTrackBuffer,
    associate_face_to_person,
    STATE_CONFIRMED,
)
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from geometry_msgs.msg import Point32
from vision_msgs.msg import BoundingBox2D, Pose2D
from sensor_msgs.msg import Image
from thor_msgs.msg import FaceTracks, FaceTrack, FaceLandmarks, PerceptionMetrics, PersonTracks
from thor_msgs.srv import GetCameraFrame
from cv_bridge import CvBridge
from thor_telemetry import get_logger as _get_structured_logger, ErrorCode

_structured_logger = _get_structured_logger("face_detection_node")


# =============================================================================
# Configuration
# =============================================================================

CPU_FALLBACK_ENV_VAR = "FACE_PIPELINE_ALLOW_CPU_FALLBACK"
DEFAULT_ENGINE_PATH = "/opt/models/face/yunet.engine"
DEFAULT_ONNX_PATH = "/opt/models/face/yunet.onnx"

# Degraded mode warning interval (seconds)
DEGRADED_MODE_LOG_INTERVAL = 30.0


@dataclass
class RawFaceDetection:
    """Internal face detection before ROS message conversion."""
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float
    landmarks: np.ndarray  # (10,) array: [left_eye_x, left_eye_y, right_eye_x, ...]


class FaceDetectorProtocol(Protocol):
    """Protocol for face detector implementations."""
    def detect(self, image: np.ndarray) -> list[RawFaceDetection]: ...


# =============================================================================
# YuNet Post-Processing (shared between TensorRT and ONNX backends)
# =============================================================================

class YuNetPostProcessor:
    """Shared post-processing logic for YuNet YOLO-style outputs.

    This uses the correct YOLO-style decode where:
    - cx = (col + tx) * stride (NOT prior_cx + tx * stride)
    - cy = (row + ty) * stride
    - score = sqrt(cls * obj)

    Reference: OpenCV's face_detect.cpp implementation.
    """

    def __init__(self, input_size: tuple = (640, 640),
                 conf_threshold: float = 0.6, nms_threshold: float = 0.3):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.strides = [8, 16, 32]

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Preprocess image for YuNet inference with center letterboxing.

        Returns:
            (blob, scale, pad_x, pad_y) - blob for inference, scale factor, and padding offsets
        """
        h, w = image.shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Center letterbox padding
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        blob = padded.astype(np.float32)
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # Add batch dim
        return blob, scale, pad_x, pad_y

    def decode_outputs(self, outputs_dict: dict, scale: float,
                        pad_x: int = 0, pad_y: int = 0) -> list[RawFaceDetection]:
        """Decode YuNet YOLO-style outputs into face detections.

        Args:
            outputs_dict: Dict with keys like 'cls_8', 'obj_8', 'bbox_8', 'kps_8' per stride
            scale: Preprocessing scale factor
            pad_x: Horizontal padding offset (for center letterbox)
            pad_y: Vertical padding offset (for center letterbox)
        """
        input_w, input_h = self.input_size
        all_detections = []

        for stride in self.strides:
            cls_key = f'cls_{stride}'
            obj_key = f'obj_{stride}'
            bbox_key = f'bbox_{stride}'
            kps_key = f'kps_{stride}'

            # Check if per-stride keys exist, otherwise use concatenated format
            if cls_key not in outputs_dict:
                # Fall back to old concatenated format for backward compatibility
                return self._decode_concatenated(outputs_dict, scale, pad_x, pad_y)

            cls = outputs_dict[cls_key].reshape(-1).astype(np.float32)
            obj = outputs_dict[obj_key].reshape(-1).astype(np.float32)
            bbox = outputs_dict[bbox_key].reshape(-1, 4).astype(np.float32)
            kps = outputs_dict[kps_key].reshape(-1, 10).astype(np.float32)

            cols = input_w // stride
            rows = input_h // stride
            N = rows * cols

            # Score fusion: sqrt(cls * obj) per OpenCV implementation
            cls = np.clip(cls, 0.0, 1.0)
            obj = np.clip(obj, 0.0, 1.0)
            scores = np.sqrt(cls * obj)

            # Filter by confidence threshold
            keep = scores >= self.conf_threshold
            if not np.any(keep):
                continue

            scores_k = scores[keep]
            bbox_k = bbox[keep]
            kps_k = kps[keep]

            # Get grid indices (row, col) for kept detections
            idx = np.flatnonzero(keep)
            row = (idx // cols).astype(np.float32)
            col = (idx % cols).astype(np.float32)

            # YOLO-style bbox decode: cx = (col + tx) * stride
            tx, ty, tw, th = bbox_k[:, 0], bbox_k[:, 1], bbox_k[:, 2], bbox_k[:, 3]
            cx = (col + tx) * stride
            cy = (row + ty) * stride
            w = np.exp(tw) * stride
            h = np.exp(th) * stride

            # YOLO-style keypoint decode: kx = (kps_x + col) * stride
            kps_xy = kps_k.reshape(-1, 5, 2)
            kps_xy[..., 0] = (kps_xy[..., 0] + col[:, None]) * stride
            kps_xy[..., 1] = (kps_xy[..., 1] + row[:, None]) * stride

            # De-letterbox: remove padding and scale to original image
            cx_orig = (cx - pad_x) / scale
            cy_orig = (cy - pad_y) / scale
            w_orig = w / scale
            h_orig = h / scale

            kps_orig = kps_xy.copy()
            kps_orig[..., 0] = (kps_xy[..., 0] - pad_x) / scale
            kps_orig[..., 1] = (kps_xy[..., 1] - pad_y) / scale

            # Create detections
            for i in range(len(scores_k)):
                kps_flat = kps_orig[i].flatten()
                # Reorder: YuNet outputs [right_eye, left_eye, nose, right_mouth, left_mouth]
                # to: [left_eye, right_eye, nose, left_mouth, right_mouth]
                landmarks = np.array([
                    kps_flat[2], kps_flat[3],  # left_eye
                    kps_flat[0], kps_flat[1],  # right_eye
                    kps_flat[4], kps_flat[5],  # nose
                    kps_flat[8], kps_flat[9],  # left_mouth
                    kps_flat[6], kps_flat[7],  # right_mouth
                ])

                all_detections.append(RawFaceDetection(
                    center_x=float(cx_orig[i]),
                    center_y=float(cy_orig[i]),
                    width=float(w_orig[i]),
                    height=float(h_orig[i]),
                    confidence=float(scores_k[i]),
                    landmarks=landmarks,
                ))

        if not all_detections:
            return []

        # NMS across all strides
        boxes = np.array([[d.center_x - d.width/2, d.center_y - d.height/2,
                          d.width, d.height] for d in all_detections])
        scores = np.array([d.confidence for d in all_detections])

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.conf_threshold, self.nms_threshold
        )

        if len(indices) == 0:
            return []

        return [all_detections[i] for i in indices.flatten()]

    def _decode_concatenated(self, outputs_dict: dict, scale: float,
                              pad_x: int, pad_y: int) -> list[RawFaceDetection]:
        """Fallback decoder for old concatenated output format."""
        # This handles the case where outputs are pre-concatenated
        # (for backward compatibility with existing TensorRT code)
        cls_scores = outputs_dict.get('cls', np.array([])).flatten()
        obj_scores = outputs_dict.get('obj', np.ones_like(cls_scores))
        bbox_deltas = outputs_dict.get('bbox', np.array([]))
        kps_deltas = outputs_dict.get('kps', np.array([]))

        if len(cls_scores) == 0:
            return []

        # Use sqrt(cls * obj) scoring
        cls_scores = np.clip(cls_scores, 0.0, 1.0)
        obj_scores = np.clip(obj_scores.flatten(), 0.0, 1.0)
        scores = np.sqrt(cls_scores * obj_scores)

        input_w, input_h = self.input_size

        # Build grid indices for all strides
        all_rows = []
        all_cols = []
        all_strides = []
        for stride in self.strides:
            cols = input_w // stride
            rows = input_h // stride
            for r in range(rows):
                for c in range(cols):
                    all_rows.append(r)
                    all_cols.append(c)
                    all_strides.append(stride)

        all_rows = np.array(all_rows, dtype=np.float32)
        all_cols = np.array(all_cols, dtype=np.float32)
        all_strides = np.array(all_strides, dtype=np.float32)

        # Filter by confidence
        mask = scores > self.conf_threshold
        if not np.any(mask):
            return []

        scores = scores[mask]
        bbox_deltas = bbox_deltas[mask]
        kps_deltas = kps_deltas[mask]
        rows = all_rows[mask]
        cols = all_cols[mask]
        strides = all_strides[mask]

        # YOLO-style decode
        tx, ty, tw, th = bbox_deltas[:, 0], bbox_deltas[:, 1], bbox_deltas[:, 2], bbox_deltas[:, 3]
        cx = (cols + tx) * strides
        cy = (rows + ty) * strides
        w = np.exp(tw) * strides
        h = np.exp(th) * strides

        # Keypoints
        kps_xy = kps_deltas.reshape(-1, 5, 2)
        kps_xy[..., 0] = (kps_xy[..., 0] + cols[:, None]) * strides[:, None]
        kps_xy[..., 1] = (kps_xy[..., 1] + rows[:, None]) * strides[:, None]

        # De-letterbox
        cx_orig = (cx - pad_x) / scale
        cy_orig = (cy - pad_y) / scale
        w_orig = w / scale
        h_orig = h / scale
        kps_orig = kps_xy.copy()
        kps_orig[..., 0] = (kps_xy[..., 0] - pad_x) / scale
        kps_orig[..., 1] = (kps_xy[..., 1] - pad_y) / scale

        # NMS
        boxes = np.stack([cx_orig - w_orig/2, cy_orig - h_orig/2, w_orig, h_orig], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.conf_threshold, self.nms_threshold
        )

        if len(indices) == 0:
            return []

        detections = []
        for idx in indices.flatten():
            kps_flat = kps_orig[idx].flatten()
            landmarks = np.array([
                kps_flat[2], kps_flat[3], kps_flat[0], kps_flat[1], kps_flat[4], kps_flat[5],
                kps_flat[8], kps_flat[9], kps_flat[6], kps_flat[7],
            ])
            detections.append(RawFaceDetection(
                center_x=float(cx_orig[idx]),
                center_y=float(cy_orig[idx]),
                width=float(w_orig[idx]),
                height=float(h_orig[idx]),
                confidence=float(scores[idx]),
                landmarks=landmarks,
            ))

        return detections


# =============================================================================
# TensorRT Backend (Default)
# =============================================================================

class YuNetTensorRT:
    """YuNet face detector using TensorRT engine."""

    def __init__(self, engine_path: str, input_size: tuple = (640, 640),
                 conf_threshold: float = 0.6, nms_threshold: float = 0.3,
                 logger=None):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 - Required for CUDA context

        self.logger = logger
        self.postprocessor = YuNetPostProcessor(input_size, conf_threshold, nms_threshold)

        # Load TensorRT engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            device_mem = cuda.mem_alloc(size)
            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape})
                self.output_names.append(name)

        self.stream = cuda.Stream()

        if logger:
            logger.info(f"TensorRT engine loaded: {engine_path}")
            logger.info(f"  Outputs: {self.output_names}")

    def detect(self, image: np.ndarray) -> list[RawFaceDetection]:
        """Run face detection using TensorRT."""
        import pycuda.driver as cuda

        blob, scale, pad_x, pad_y = self.postprocessor.preprocess(image)

        # Copy input to device
        np.copyto(self.inputs[0]['host'], blob.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs back
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

        # Gather outputs preserving per-stride structure (cls_8, obj_8, bbox_8, kps_8, etc.)
        outputs_dict = {}
        for out in self.outputs:
            name = out['name']
            data = out['host'].reshape(out['shape'])
            outputs_dict[name] = data

        return self.postprocessor.decode_outputs(outputs_dict, scale, pad_x, pad_y)


# =============================================================================
# ONNX Runtime Backend (CPU Fallback - Disabled by Default)
# =============================================================================

class YuNetONNX:
    """YuNet face detector using ONNX Runtime (CPU fallback)."""

    def __init__(self, model_path: str, input_size: tuple = (640, 640),
                 conf_threshold: float = 0.6, nms_threshold: float = 0.3,
                 logger=None):
        import onnxruntime as ort

        self.logger = logger
        self.postprocessor = YuNetPostProcessor(input_size, conf_threshold, nms_threshold)

        # Force CPU provider only for degraded mode
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        if logger:
            logger.warn("=" * 60)
            logger.warn("DEGRADED MODE: Using ONNX Runtime CPU inference")
            logger.warn("Performance will be significantly reduced!")
            logger.warn("Build TensorRT engine: build_face_engine.sh")
            logger.warn("=" * 60)

    def detect(self, image: np.ndarray) -> list[RawFaceDetection]:
        """Run face detection using ONNX Runtime."""
        blob, scale, pad_x, pad_y = self.postprocessor.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        # Preserve per-stride structure (cls_8, obj_8, bbox_8, kps_8, etc.)
        outputs_dict = {name: output for name, output in zip(self.output_names, outputs)}

        return self.postprocessor.decode_outputs(outputs_dict, scale, pad_x, pad_y)


# =============================================================================
# Detector Factory
# =============================================================================

def create_detector(engine_path: str, onnx_path: str, input_size: tuple,
                    conf_threshold: float, nms_threshold: float,
                    logger) -> tuple[FaceDetectorProtocol, bool]:
    """Create face detector with TensorRT default, guarded CPU fallback.

    Returns:
        (detector, is_degraded_mode)
    """
    # Check if TensorRT engine exists
    if Path(engine_path).exists():
        try:
            detector = YuNetTensorRT(
                engine_path, input_size, conf_threshold, nms_threshold, logger
            )
            logger.info("Face detection: TensorRT GPU inference (optimal)")
            return detector, False
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")

    # TensorRT not available - check if CPU fallback allowed
    allow_cpu = os.environ.get(CPU_FALLBACK_ENV_VAR, "").lower() == "true"

    if not allow_cpu:
        logger.fatal("=" * 70)
        logger.fatal("FATAL: TensorRT engine not found and CPU fallback disabled")
        logger.fatal(f"  Engine path: {engine_path}")
        logger.fatal("")
        logger.fatal("To fix, run ONCE after deployment:")
        logger.fatal("  /opt/ros2_ws/src/thor_perception/scripts/build_face_engine.sh")
        logger.fatal("")
        logger.fatal(f"Or enable CPU fallback (NOT recommended for production):")
        logger.fatal(f"  export {CPU_FALLBACK_ENV_VAR}=true")
        logger.fatal("=" * 70)
        raise RuntimeError(
            f"TensorRT engine missing at {engine_path}. "
            f"Run build_face_engine.sh or set {CPU_FALLBACK_ENV_VAR}=true"
        )

    # CPU fallback enabled - use ONNX Runtime with loud warnings
    if not Path(onnx_path).exists():
        raise RuntimeError(f"Neither TensorRT engine nor ONNX model found")

    detector = YuNetONNX(onnx_path, input_size, conf_threshold, nms_threshold, logger)
    return detector, True


# =============================================================================
# ROS Node
# =============================================================================

class FaceDetectionNode(Node):
    """ROS 2 node that runs YuNet face detection and publishes FaceTracks."""

    def __init__(self):
        super().__init__("face_detection_node")

        # Declare parameters
        self.declare_parameter("engine_path", DEFAULT_ENGINE_PATH)
        self.declare_parameter("onnx_path", DEFAULT_ONNX_PATH)
        self.declare_parameter("input_size", 640)
        self.declare_parameter("min_confidence", 0.6)
        self.declare_parameter("nms_threshold", 0.3)
        self.declare_parameter("min_face_size_px", 40)
        self.declare_parameter("quality_min_confidence", 0.7)
        self.declare_parameter("quality_min_size_px", 60)
        self.declare_parameter("max_publish_rate_hz", 10.0)  # Reduced from 15 for identity gate
        self.declare_parameter("publish_metrics", True)
        self.declare_parameter("metrics_interval_s", 5.0)
        self.declare_parameter("min_fps_threshold", float(os.environ.get("PERCEPTION_MIN_FPS", "5.0")))
        self.declare_parameter("camera_id", "camera0")
        self.declare_parameter("camera_hfov_deg", 65.0)
        self.declare_parameter("publish_frames", True)  # For downstream face ID

        # Crop buffer parameters (for on-demand AuraFace inference)
        self.declare_parameter("crop_buffer_enabled", True)
        self.declare_parameter("crop_save_interval_sec", 0.5)  # Rate limit per track
        self.declare_parameter("crop_cleanup_interval_sec", 10.0)

        # Face-to-person association parameters (Phase 3.2)
        self.declare_parameter("association.margin_threshold", 0.15)
        self.declare_parameter("association.timestamp_tolerance_sec", 0.15)
        self.declare_parameter("association.buffer_duration_sec", 0.5)
        self.declare_parameter("association.area_ratio_min", 0.02)
        self.declare_parameter("association.area_ratio_max", 0.30)
        self.declare_parameter("association.min_score", 0.3)
        self.declare_parameter("association.require_confirmed", True)

        # Tracker parameters
        self.declare_parameter("tracker_iou_threshold", 0.3)
        self.declare_parameter("tracker_dist_threshold", 0.15)
        self.declare_parameter("tracker_size_ratio_threshold", 0.65)
        self.declare_parameter("tracker_max_missed", 10)
        self.declare_parameter("tracker_max_missed_confirmed", 15)
        self.declare_parameter("tracker_min_hits", 3)
        self.declare_parameter("tracker_cost_weight_iou", 0.6)
        self.declare_parameter("tracker_cost_weight_dist", 0.4)
        self.declare_parameter("tracker_vmax", 320.0)

        # Get parameters
        self.engine_path = self.get_parameter("engine_path").value
        self.onnx_path = self.get_parameter("onnx_path").value
        self.input_size = self.get_parameter("input_size").value
        self.min_confidence = self.get_parameter("min_confidence").value
        self.nms_threshold = self.get_parameter("nms_threshold").value
        self.min_face_size_px = self.get_parameter("min_face_size_px").value
        self.quality_min_confidence = self.get_parameter("quality_min_confidence").value
        self.quality_min_size_px = self.get_parameter("quality_min_size_px").value
        self.max_publish_rate_hz = self.get_parameter("max_publish_rate_hz").value
        self.publish_metrics = self.get_parameter("publish_metrics").value
        self.metrics_interval_s = self.get_parameter("metrics_interval_s").value
        self.min_fps_threshold = self.get_parameter("min_fps_threshold").value
        self.camera_id = self.get_parameter("camera_id").value
        self.camera_hfov_deg = self.get_parameter("camera_hfov_deg").value
        self.publish_frames = self.get_parameter("publish_frames").value

        # Crop buffer parameters
        self.crop_buffer_enabled = self.get_parameter("crop_buffer_enabled").value
        self.crop_save_interval_sec = self.get_parameter("crop_save_interval_sec").value
        self.crop_cleanup_interval_sec = self.get_parameter("crop_cleanup_interval_sec").value

        # Association parameters (Phase 3.2)
        self.assoc_margin_threshold = self.get_parameter("association.margin_threshold").value
        self.assoc_timestamp_tolerance = self.get_parameter("association.timestamp_tolerance_sec").value
        self.assoc_buffer_duration = self.get_parameter("association.buffer_duration_sec").value
        self.assoc_area_ratio_min = self.get_parameter("association.area_ratio_min").value
        self.assoc_area_ratio_max = self.get_parameter("association.area_ratio_max").value
        self.assoc_min_score = self.get_parameter("association.min_score").value
        self.assoc_require_confirmed = self.get_parameter("association.require_confirmed").value

        # Tracker parameters
        tracker_iou_threshold = self.get_parameter("tracker_iou_threshold").value
        tracker_dist_threshold = self.get_parameter("tracker_dist_threshold").value
        tracker_size_ratio_threshold = self.get_parameter("tracker_size_ratio_threshold").value
        tracker_max_missed = self.get_parameter("tracker_max_missed").value
        tracker_max_missed_confirmed = self.get_parameter("tracker_max_missed_confirmed").value
        tracker_min_hits = self.get_parameter("tracker_min_hits").value
        tracker_cost_weight_iou = self.get_parameter("tracker_cost_weight_iou").value
        tracker_cost_weight_dist = self.get_parameter("tracker_cost_weight_dist").value
        tracker_vmax = self.get_parameter("tracker_vmax").value

        # Debug tracking mode
        self.debug_tracking = os.environ.get("DEBUG_TRACKING", "") == "1"

        # Publishers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.tracks_pub = self.create_publisher(FaceTracks, "/perception/faces/tracks", qos)

        # Frame publisher for downstream face ID (best-effort, drop if consumer can't keep up)
        if self.publish_frames:
            frame_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,  # Drop frames if face ID node can't keep up
            )
            self.frame_pub = self.create_publisher(Image, "/perception/camera/frame", frame_qos)
            self.cv_bridge = CvBridge()
        else:
            self.frame_pub = None
            self.cv_bridge = None

        if self.publish_metrics:
            self.metrics_pub = self.create_publisher(PerceptionMetrics, "/perception/faces/metrics", 10)
            self.metrics_timer = self.create_timer(self.metrics_interval_s, self._publish_metrics)

        # Metrics tracking
        self.frame_times = deque(maxlen=100)
        self.latencies = deque(maxlen=100)
        self.total_faces_detected = 0
        self.total_frames_processed = 0
        self.dropped_frames_total = 0

        # Rate limiting
        self.min_publish_interval = 1.0 / self.max_publish_rate_hz if self.max_publish_rate_hz > 0 else 0
        self.last_publish_time = 0.0

        # Degraded mode tracking
        self.is_degraded_mode = False
        self.last_degraded_warning = 0.0

        # Initialize IoU tracker
        self.tracker = IoUTracker(
            iou_thresh=tracker_iou_threshold,
            dist_thresh=tracker_dist_threshold,
            size_ratio_thresh=tracker_size_ratio_threshold,
            max_missed=tracker_max_missed,
            max_missed_confirmed=tracker_max_missed_confirmed,
            min_hits=tracker_min_hits,
            cost_weight_iou=tracker_cost_weight_iou,
            cost_weight_dist=tracker_cost_weight_dist,
            vmax=tracker_vmax,
            frame_width=640,  # Will be updated when capture initializes
            frame_height=480,
        )
        self.tracker_min_hits = tracker_min_hits  # Store for quality gating

        # Initialize detector
        self.detector: Optional[FaceDetectorProtocol] = None
        self.frame_width = 640
        self.frame_height = 480

        # Image subscription (replaces RTSP capture)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # cv_bridge for image conversion (always needed now)
        if self.cv_bridge is None:
            self.cv_bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            "/sensors/camera/head/rgb/image",
            self._image_callback,
            sensor_qos,
        )

        # Person track buffer for time-coherent association (Phase 3.2)
        self.person_track_buffer = PersonTrackBuffer(
            max_age_sec=self.assoc_buffer_duration
        )
        self.person_tracks_sub = self.create_subscription(
            PersonTracks,
            "/tracking/person_tracks",
            self._person_tracks_callback,
            sensor_qos,
        )

        # Crop buffer for on-demand face ID (shared volume with thor_pipeline)
        self.crop_buffer: Optional[FaceCropBuffer] = None
        self.track_last_crop_time: dict[int, float] = {}  # Rate limit per track
        self.previous_track_ids: set[int] = set()  # For detecting retired tracks

        # Track first seen timestamps (for FaceTrack.first_seen)
        self.track_first_seen: dict[int, float] = {}

        # Frame buffer for debug snapshot endpoint
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_time: Optional[rclpy.time.Time] = None
        self._frame_lock = threading.Lock()

        # GetCameraFrame service for debug snapshots
        self.frame_service = self.create_service(
            GetCameraFrame,
            "/perception/get_camera_frame",
            self._handle_get_camera_frame,
        )

        self.get_logger().info("Face detection node initialized (ROS image subscription)")

    def _init_detector(self) -> bool:
        """Initialize YuNet detector."""
        try:
            self.detector, self.is_degraded_mode = create_detector(
                self.engine_path, self.onnx_path,
                (self.input_size, self.input_size),
                self.min_confidence, self.nms_threshold,
                self.get_logger(),
            )
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to initialize detector: {e}")
            _structured_logger.emit_failure(
                trace_ctx=None,  # CP-006 §5.1: perception has no turn-level trace
                operation="init_detector",
                error_code=ErrorCode.UNAVAILABLE,
                error_detail=str(e)[:200],
                trigger=type(e).__name__,
            )
            return False

    def _compute_bearing(self, center_x: float) -> float:
        """Compute horizontal bearing from camera center."""
        hfov_rad = self.camera_hfov_deg * math.pi / 180.0
        fx_pixels = (self.frame_width / 2) / math.tan(hfov_rad / 2)
        cx_pixels = self.frame_width / 2
        bearing_rad = math.atan2(center_x - cx_pixels, fx_pixels)
        return bearing_rad * 180.0 / math.pi

    def _compute_quality(self, det: RawFaceDetection) -> tuple[bool, str, int]:
        """Compute quality metrics for a face detection."""
        flags = 0
        reasons = []

        if det.confidence < self.quality_min_confidence:
            flags |= FaceTrack.LOW_CONFIDENCE
            reasons.append(f"conf={det.confidence:.2f}<{self.quality_min_confidence}")

        face_size = min(det.width, det.height)
        if face_size < self.quality_min_size_px:
            flags |= FaceTrack.TOO_SMALL
            reasons.append(f"size={face_size:.0f}<{self.quality_min_size_px}")

        quality_ok = flags == 0
        quality_reason = "; ".join(reasons) if reasons else "ok"
        return quality_ok, quality_reason, flags

    def _detection_to_msg(self, det: RawFaceDetection, track_id: int,
                          now: rclpy.time.Time) -> FaceTrack:
        """Convert internal detection to ROS message."""
        msg = FaceTrack()
        msg.track_id = track_id

        now_msg = now.to_msg()
        if track_id in self.track_first_seen:
            first_seen_sec = self.track_first_seen[track_id]
            msg.first_seen.sec = int(first_seen_sec)
            msg.first_seen.nanosec = int((first_seen_sec % 1) * 1e9)
        else:
            msg.first_seen = now_msg
            self.track_first_seen[track_id] = now.nanoseconds / 1e9
        msg.last_seen = now_msg

        msg.bbox.center.position.x = det.center_x
        msg.bbox.center.position.y = det.center_y
        msg.bbox.center.theta = 0.0
        msg.bbox.size_x = det.width
        msg.bbox.size_y = det.height

        lm = det.landmarks
        msg.landmarks.format = FaceLandmarks.YUNET_5PT
        msg.landmarks.left_eye_x = lm[0]
        msg.landmarks.left_eye_y = lm[1]
        msg.landmarks.right_eye_x = lm[2]
        msg.landmarks.right_eye_y = lm[3]
        msg.landmarks.nose_tip_x = lm[4]
        msg.landmarks.nose_tip_y = lm[5]
        msg.landmarks.left_mouth_x = lm[6]
        msg.landmarks.left_mouth_y = lm[7]
        msg.landmarks.right_mouth_x = lm[8]
        msg.landmarks.right_mouth_y = lm[9]

        msg.landmarks.points = [
            Point32(x=lm[0], y=lm[1], z=0.0),
            Point32(x=lm[2], y=lm[3], z=0.0),
            Point32(x=lm[4], y=lm[5], z=0.0),
            Point32(x=lm[6], y=lm[7], z=0.0),
            Point32(x=lm[8], y=lm[9], z=0.0),
        ]
        msg.landmarks.confidence = [det.confidence] * 5

        msg.detector_confidence = det.confidence
        msg.face_size_px = min(det.width, det.height)

        quality_ok, quality_reason, quality_flags = self._compute_quality(det)
        msg.quality_ok = quality_ok
        msg.quality_reason = quality_reason
        msg.quality_flags = quality_flags

        msg.detection_count = 1
        first_seen_ts = self.track_first_seen.get(track_id, now.nanoseconds / 1e9)
        msg.age_ms = int((now.nanoseconds / 1e9 - first_seen_ts) * 1000)
        msg.last_seen_ms = 0
        msg.stability_score = det.confidence

        msg.source_frame_id = self.camera_id
        msg.source_width = self.frame_width
        msg.source_height = self.frame_height
        msg.bearing_deg = self._compute_bearing(det.center_x)

        msg.associated_person_track_id = 0
        msg.association_confidence = 0.0

        return msg

    def _tracked_face_to_msg(self, track: TrackedFace, now: rclpy.time.Time) -> FaceTrack:
        """Convert TrackedFace to ROS FaceTrack message."""
        msg = FaceTrack()
        msg.track_id = track.track_id

        # Timestamps
        msg.first_seen.sec = int(track.first_seen)
        msg.first_seen.nanosec = int((track.first_seen % 1) * 1e9)
        msg.last_seen.sec = int(track.last_seen)
        msg.last_seen.nanosec = int((track.last_seen % 1) * 1e9)

        # Bounding box (center + size format)
        msg.bbox.center.position.x = track.cx
        msg.bbox.center.position.y = track.cy
        msg.bbox.center.theta = 0.0
        msg.bbox.size_x = track.w
        msg.bbox.size_y = track.h

        # Landmarks
        lm = track.landmarks
        msg.landmarks.format = FaceLandmarks.YUNET_5PT
        msg.landmarks.left_eye_x = float(lm[0])
        msg.landmarks.left_eye_y = float(lm[1])
        msg.landmarks.right_eye_x = float(lm[2])
        msg.landmarks.right_eye_y = float(lm[3])
        msg.landmarks.nose_tip_x = float(lm[4])
        msg.landmarks.nose_tip_y = float(lm[5])
        msg.landmarks.left_mouth_x = float(lm[6])
        msg.landmarks.left_mouth_y = float(lm[7])
        msg.landmarks.right_mouth_x = float(lm[8])
        msg.landmarks.right_mouth_y = float(lm[9])

        msg.landmarks.points = [
            Point32(x=float(lm[0]), y=float(lm[1]), z=0.0),
            Point32(x=float(lm[2]), y=float(lm[3]), z=0.0),
            Point32(x=float(lm[4]), y=float(lm[5]), z=0.0),
            Point32(x=float(lm[6]), y=float(lm[7]), z=0.0),
            Point32(x=float(lm[8]), y=float(lm[9]), z=0.0),
        ]
        msg.landmarks.confidence = [track.confidence] * 5

        # Quality metrics
        msg.detector_confidence = track.confidence
        msg.face_size_px = track.face_size

        # Quality gating
        flags = 0
        reasons = []

        if track.confidence < self.quality_min_confidence:
            flags |= FaceTrack.LOW_CONFIDENCE
            reasons.append(f"conf={track.confidence:.2f}<{self.quality_min_confidence}")

        if track.face_size < self.quality_min_size_px:
            flags |= FaceTrack.TOO_SMALL
            reasons.append(f"size={track.face_size:.0f}<{self.quality_min_size_px}")

        # TOO_NEW flag for unconfirmed tracks
        if track.detection_count < self.tracker_min_hits:
            flags |= FaceTrack.TOO_NEW
            reasons.append(f"count={track.detection_count}<{self.tracker_min_hits}")

        msg.quality_ok = flags == 0
        msg.quality_reason = "; ".join(reasons) if reasons else "ok"
        msg.quality_flags = flags

        # Tracking state
        msg.detection_count = track.detection_count
        msg.age_ms = int((track.last_seen - track.first_seen) * 1000)
        msg.last_seen_ms = int((time.monotonic() - track.last_seen) * 1000)
        msg.stability_score = track.stability_score

        # Source frame info
        msg.source_frame_id = self.camera_id
        msg.source_width = self.frame_width
        msg.source_height = self.frame_height
        msg.bearing_deg = self._compute_bearing(track.cx)

        # Person association (populated in _process_frame)
        msg.associated_person_track_id = 0
        msg.association_confidence = 0.0

        return msg

    def _person_tracks_callback(self, msg: PersonTracks) -> None:
        """Add PersonTracks to buffer for time-coherent association."""
        self.person_track_buffer.add(msg)

    def _image_callback(self, msg: Image) -> None:
        """Process incoming image from camera_ingest."""
        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
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

        # Update frame dimensions if changed
        h, w = frame.shape[:2]
        if w != self.frame_width or h != self.frame_height:
            self.frame_width = w
            self.frame_height = h
            self.tracker.set_frame_size(w, h)
            self.get_logger().info(f"Frame size updated: {w}x{h}")

        # Get image timestamp for time-coherent association
        image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Get closest PersonTracks snapshot (thread-safe)
        person_tracks_msg = self.person_track_buffer.get_closest(
            image_stamp, self.assoc_timestamp_tolerance
        )

        # Process frame with time-coherent person tracks
        self._process_frame(frame, msg.header, person_tracks_msg)

    def _log_degraded_warning(self):
        """Emit periodic warnings when in degraded mode."""
        if not self.is_degraded_mode:
            return
        now = time.time()
        if now - self.last_degraded_warning > DEGRADED_MODE_LOG_INTERVAL:
            self.get_logger().warn("=" * 50)
            self.get_logger().warn("RUNNING IN DEGRADED MODE: CPU INFERENCE")
            self.get_logger().warn(f"FPS: {self._calculate_fps():.1f} (expect <5 FPS)")
            self.get_logger().warn("Run build_face_engine.sh to fix")
            self.get_logger().warn("=" * 50)
            self.last_degraded_warning = now

    def _handle_get_camera_frame(self, request, response):
        """Handle GetCameraFrame service request."""
        with self._frame_lock:
            if self._latest_frame is None:
                response.success = False
                response.error = "NO_FRAME_AVAILABLE"
                return response

            frame = self._latest_frame.copy()
            frame_time = self._latest_frame_time

        # Encode as JPEG
        quality = request.quality if request.quality > 0 else 80
        quality = max(1, min(100, quality))

        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success, jpeg_data = cv2.imencode('.jpg', frame, encode_params)

            if not success:
                response.success = False
                response.error = "JPEG_ENCODE_FAILED"
                return response

            response.success = True
            response.error = ""
            response.width = frame.shape[1]
            response.height = frame.shape[0]
            response.timestamp = frame_time.to_msg() if frame_time else self.get_clock().now().to_msg()
            response.jpeg_data = jpeg_data.tobytes()

        except Exception as e:
            response.success = False
            response.error = f"ENCODE_ERROR: {str(e)}"

        return response

    def _process_frame(self, frame: np.ndarray, header: Header,
                       person_tracks_msg: Optional[PersonTracks]) -> None:
        """Process a single frame and publish detections.

        Args:
            frame: BGR image from camera_ingest
            header: Original image header (MUST use header.stamp, not now())
            person_tracks_msg: Time-coherent PersonTracks for association (may be None)
        """
        start_time = time.monotonic()
        # Use image timestamp for all message timestamps (critical for replay determinism)
        now = rclpy.time.Time(
            seconds=header.stamp.sec,
            nanoseconds=header.stamp.nanosec,
            clock_type=self.get_clock().clock_type
        )

        # Store latest frame for debug snapshot service
        with self._frame_lock:
            self._latest_frame = frame.copy()
            self._latest_frame_time = now

        # Run YuNet detection
        raw_detections = self.detector.detect(frame)
        raw_detections = [d for d in raw_detections if min(d.width, d.height) >= self.min_face_size_px]

        # Convert to tracker format
        tracker_detections = [
            RawDetection(
                cx=d.center_x,
                cy=d.center_y,
                w=d.width,
                h=d.height,
                confidence=d.confidence,
                landmarks=d.landmarks,
            )
            for d in raw_detections
        ]

        # Run tracker
        current_time = time.monotonic()
        tracked_faces, matches = self.tracker.update(tracker_detections, current_time, debug=self.debug_tracking)

        # Handle crop buffer: save crops for quality tracks, retire lost tracks
        if self.crop_buffer is not None:
            current_track_ids = {t.track_id for t in tracked_faces}

            # Retire lost tracks (delete their crops)
            retired_ids = self.previous_track_ids - current_track_ids
            for track_id in retired_ids:
                self.crop_buffer.retire_track(track_id)
                self.track_last_crop_time.pop(track_id, None)
                self.track_first_seen.pop(track_id, None)

            # Save crops for quality tracks (rate limited per track)
            for track in tracked_faces:
                # Skip unconfirmed or low-quality tracks
                if track.detection_count < self.tracker_min_hits:
                    continue
                if track.face_size < self.quality_min_size_px:
                    continue
                if track.confidence < self.quality_min_confidence:
                    continue

                # Rate limit: skip if we saved a crop recently for this track
                last_crop = self.track_last_crop_time.get(track.track_id, 0.0)
                if current_time - last_crop < self.crop_save_interval_sec:
                    continue

                # Compute bbox (x, y, w, h) from center format
                x = int(track.cx - track.w / 2)
                y = int(track.cy - track.h / 2)
                w = int(track.w)
                h = int(track.h)

                # Save crop
                crop_path = self.crop_buffer.add_crop(
                    track_id=track.track_id,
                    frame=frame,
                    bbox=(x, y, w, h),
                    quality_flags=[],
                )
                if crop_path:
                    self.track_last_crop_time[track.track_id] = current_time

            self.previous_track_ids = current_track_ids

        # Debug logging
        if self.debug_tracking and matches:
            for m in matches:
                self.get_logger().debug(
                    f"MATCH: track={m.track_id} <- det={m.det_idx} "
                    f"cost={m.cost:.3f} iou={m.iou:.2f} dist={m.dist:.3f}"
                )

        # Rate limiting
        if current_time - self.last_publish_time < self.min_publish_interval:
            return
        self.last_publish_time = current_time

        # Build message (use original image header for timestamp)
        msg = FaceTracks()
        msg.header = header  # CRITICAL: Use image timestamp, not now()

        # Extract person tracks for association
        person_tracks = person_tracks_msg.tracks if person_tracks_msg else []

        for track in tracked_faces:
            track_msg = self._tracked_face_to_msg(track, now)

            # Run face-to-person association (Phase 3.2)
            if person_tracks:
                result = associate_face_to_person(
                    face_cx=track.cx,
                    face_cy=track.cy,
                    face_w=track.w,
                    face_h=track.h,
                    person_tracks=person_tracks,
                    require_confirmed=self.assoc_require_confirmed,
                    area_ratio_min=self.assoc_area_ratio_min,
                    area_ratio_max=self.assoc_area_ratio_max,
                    margin_threshold=self.assoc_margin_threshold,
                    min_score=self.assoc_min_score,
                )
                track_msg.associated_person_track_id = result.person_track_id
                track_msg.association_confidence = result.confidence

            msg.tracks.append(track_msg)

        process_time = time.monotonic() - start_time
        self.frame_times.append(current_time)
        self.latencies.append(process_time * 1000)
        self.total_faces_detected += len(raw_detections)
        self.total_frames_processed += 1

        msg.fps = self._calculate_fps()
        msg.total_faces_detected = self.total_faces_detected

        self.tracks_pub.publish(msg)

        # Publish frame for downstream face ID (same timestamp for matching)
        if self.frame_pub is not None and self.cv_bridge is not None:
            try:
                img_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                img_msg.header = msg.header  # Same timestamp as FaceTracks
                self.frame_pub.publish(img_msg)
                # DEBUG: Log timestamp alignment (every 100 frames)
                if self.total_frames_processed % 100 == 1:
                    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    self.get_logger().info(f"FRAME_PUB: stamp={stamp:.3f} tracks={len(msg.tracks)}")
            except Exception as e:
                self.get_logger().warning(f"Frame publish failed: {e}")

        self._log_degraded_warning()

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
        msg.header.frame_id = self.camera_id

        msg.fps = self._calculate_fps()
        msg.latency_p95_ms = float(np.percentile(self.latencies, 95)) if self.latencies else 0.0
        msg.dropped_frames_total = self.dropped_frames_total
        msg.reconnect_count_total = 0
        msg.queue_depth = 0

        msg.model_name = "YuNet"
        msg.model_version = "2023mar"
        msg.tracker_enabled = True
        msg.reconnect_count_total = 0  # No RTSP reconnects in Phase 3.2

        # Compute health status
        if self.is_degraded_mode:
            msg.health = PerceptionMetrics.DEGRADED
            msg.reason_code = "CPU_FALLBACK"
        elif msg.fps < self.min_fps_threshold:
            msg.health = PerceptionMetrics.DEGRADED
            msg.reason_code = "LOW_FPS"
        else:
            msg.health = PerceptionMetrics.OK
            msg.reason_code = ""

        self.metrics_pub.publish(msg)

    def _cleanup_crops(self):
        """Periodic cleanup of stale crops and orphan files."""
        if self.crop_buffer is None:
            return

        try:
            stale = self.crop_buffer.cleanup_stale()
            orphans = self.crop_buffer.cleanup_orphan_files()
            stats = self.crop_buffer.get_stats()

            if stale > 0 or orphans > 0:
                self.get_logger().debug(
                    f"Crop cleanup: stale={stale} orphans={orphans} "
                    f"total={stats['total_crops']} tracks={stats['num_tracks']}"
                )
        except Exception as e:
            self.get_logger().warning(f"Crop cleanup error: {e}")

    def start(self):
        """Start the face detection pipeline.

        Phase 3.2: No longer handles RTSP directly. Images come from ROS subscription.
        """
        self.get_logger().info("Starting face detection pipeline...")

        if not self._init_detector():
            return False

        # Initialize crop buffer (doesn't depend on image subscription)
        if self.crop_buffer_enabled:
            self.crop_buffer = FaceCropBuffer(logger=self.get_logger())
            orphans = self.crop_buffer.cleanup_orphan_files()
            if orphans > 0:
                self.get_logger().info(f"Cleaned up {orphans} orphan crop files")
            self.crop_cleanup_timer = self.create_timer(
                self.crop_cleanup_interval_sec,
                self._cleanup_crops,
            )
            self.get_logger().info("Crop buffer enabled for on-demand face ID")

        mode = "DEGRADED (CPU)" if self.is_degraded_mode else "GPU (TensorRT)"
        self.get_logger().info(f"Face detection pipeline started - Mode: {mode}")
        self.get_logger().info("Waiting for images on /sensors/camera/head/rgb/image...")
        return True

    # No blocking I/O in stop — budget is zero (CP-008)
    _STOP_BUDGET_SEC = 0.0

    def stop(self):
        """Stop the face detection pipeline (budget: _STOP_BUDGET_SEC)."""
        # Clear person track buffer
        self.person_track_buffer.clear()
        self.get_logger().info("Face detection pipeline stopped")


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()

    if not node.start():
        node.get_logger().error("Failed to start face detection pipeline")
        rclpy.shutdown()
        return 1

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
