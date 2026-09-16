"""PeopleNet TensorRT inference module.

Provides GPU-accelerated person detection using NVIDIA PeopleNet model.
Follows the same patterns as YuNetTensorRT in face_detection_node.py.

Usage:
    detector = PeopleNetTRT('/opt/models/perception/peoplenet.engine')
    detections = detector.detect(bgr_image)
    for det in detections:
        print(f"Person at ({det.cx}, {det.cy}) conf={det.score:.2f}")
"""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from thor_telemetry import (
    BoundaryContract, BoundaryType, ErrorBehavior,
)

from thor_telemetry import get_logger as _get_structured_logger, ErrorCode
_structured_logger = _get_structured_logger("peoplenet_trt")

# --- CP-010 Boundary Contract (SI-11.1) ---
PEOPLENET_DETECT_CONTRACT = BoundaryContract(
    boundary_name="peoplenet_trt_detect",
    boundary_type=BoundaryType.GPU_INFERENCE,
    timeout_sec=2.0,
    error_behavior=ErrorBehavior.DEGRADE,
    retry_policy=None,
    error_codes=frozenset({ErrorCode.TIMEOUT, ErrorCode.INTERNAL}),
)


@dataclass
class Detection:
    """Person detection in original image coordinates."""

    x1: float  # Left edge in pixels
    y1: float  # Top edge in pixels
    x2: float  # Right edge in pixels
    y2: float  # Bottom edge in pixels
    score: float  # Confidence score [0, 1]
    class_id: int = 0  # Always 0 for person

    @property
    def cx(self) -> float:
        """Center X coordinate."""
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        """Center Y coordinate."""
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        """Bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Bounding box height."""
        return self.y2 - self.y1


class PeopleNetPostProcessor:
    """Decode PeopleNet outputs to detections.

    PeopleNet uses a grid-based detection scheme:
    - Input: 960x544 (W x H)
    - Grid: 60x34 (stride 16)
    - 3 classes: person=0, bag=1, face=2

    Output tensors:
    - coverage: (1, 3, 34, 60) - confidence per class per cell
    - bbox: (1, 12, 34, 60) - 4 deltas per class (x1, y1, x2, y2)
    """

    INPUT_WIDTH = 960
    INPUT_HEIGHT = 544
    STRIDE = 16
    GRID_W = 60  # 960 / 16
    GRID_H = 34  # 544 / 16

    def __init__(self, conf_threshold: float = 0.5, nms_iou_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox resize to 960x544, normalize to [0,1], BGR->RGB, HWC->CHW.

        Returns:
            (blob, scale, pad_x, pad_y)
        """
        h, w = image.shape[:2]
        target_w, target_h = self.INPUT_WIDTH, self.INPUT_HEIGHT

        # Compute scale preserving aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h))

        # Pad to target size (center padding with gray)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # Normalize to [0, 1] and convert BGR -> RGB
        blob = padded.astype(np.float32) / 255.0
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)

        # HWC -> CHW and add batch dimension
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0)

        return blob, scale, pad_x, pad_y

    def postprocess(
        self,
        coverage: np.ndarray,
        bbox: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[Detection]:
        """Decode PeopleNet outputs to detections.

        Args:
            coverage: (1, 3, 34, 60) confidence maps
            bbox: (1, 12, 34, 60) bbox deltas
            scale: Preprocessing scale factor
            pad_x: Horizontal padding applied
            pad_y: Vertical padding applied

        Returns:
            List of Detection objects in original image coordinates
        """
        # Extract person class only (index 0)
        person_cov = coverage[0, 0]  # (34, 60)
        person_bbox = bbox[0, 0:4]  # (4, 34, 60)

        # Find cells above confidence threshold
        ys, xs = np.where(person_cov > self.conf_threshold)

        if len(ys) == 0:
            return []

        # TAO DetectNet_v2 bbox decode constants
        # Reference: NVIDIA TAO bbox objective uses scale=35.0, offset=0.5
        bbox_norm = 35.0
        offset = 0.5

        detections = []
        for y, x in zip(ys, xs):
            confidence = float(person_cov[y, x])

            # Encoded values from network
            bx1, by1, bx2, by2 = person_bbox[:, y, x]

            # Grid-cell anchor terms (in "normalized-by-35" units)
            cx = (x * self.STRIDE + offset) / bbox_norm
            cy = (y * self.STRIDE + offset) / bbox_norm

            # Decode to network-input pixel coordinates (960x544 space)
            # Sign conventions per TAO DetectNet_v2 spec
            x1_pre = (bx1 - cx) * (-bbox_norm)
            y1_pre = (by1 - cy) * (-bbox_norm)
            x2_pre = (bx2 + cx) * bbox_norm
            y2_pre = (by2 + cy) * bbox_norm

            # Clip to preprocessed image bounds before inverse letterbox
            x1_pre = max(0.0, min(x1_pre, self.INPUT_WIDTH))
            y1_pre = max(0.0, min(y1_pre, self.INPUT_HEIGHT))
            x2_pre = max(0.0, min(x2_pre, self.INPUT_WIDTH))
            y2_pre = max(0.0, min(y2_pre, self.INPUT_HEIGHT))

            # Ensure x1 <= x2 and y1 <= y2
            if x1_pre > x2_pre:
                x1_pre, x2_pre = x2_pre, x1_pre
            if y1_pre > y2_pre:
                y1_pre, y2_pre = y2_pre, y1_pre

            # Remove letterbox padding and scale back to original image coordinates
            x1 = (x1_pre - pad_x) / scale
            y1 = (y1_pre - pad_y) / scale
            x2 = (x2_pre - pad_x) / scale
            y2 = (y2_pre - pad_y) / scale

            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=confidence))

        # Apply NMS
        if len(detections) > 0:
            detections = self._nms(detections)

        return detections

    def postprocess_faces(
        self,
        coverage: np.ndarray,
        bbox: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        conf_threshold: Optional[float] = None,
    ) -> list[Detection]:
        """Decode PeopleNet face detections (class 2).

        Args:
            coverage: (1, 3, 34, 60) confidence maps
            bbox: (1, 12, 34, 60) bbox deltas
            scale: Preprocessing scale factor
            pad_x: Horizontal padding applied
            pad_y: Vertical padding applied
            conf_threshold: Optional override for confidence threshold

        Returns:
            List of Detection objects for faces in original image coordinates
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        # Extract face class (index 2)
        face_cov = coverage[0, 2]  # (34, 60)
        face_bbox = bbox[0, 8:12]  # (4, 34, 60) - channels 8-11 for class 2

        # Find cells above confidence threshold
        ys, xs = np.where(face_cov > threshold)

        if len(ys) == 0:
            return []

        # TAO DetectNet_v2 bbox decode constants
        bbox_norm = 35.0
        offset = 0.5

        detections = []
        for y, x in zip(ys, xs):
            confidence = float(face_cov[y, x])

            # Encoded values from network
            bx1, by1, bx2, by2 = face_bbox[:, y, x]

            # Grid-cell anchor terms
            cx = (x * self.STRIDE + offset) / bbox_norm
            cy = (y * self.STRIDE + offset) / bbox_norm

            # Decode to network-input pixel coordinates
            x1_pre = (bx1 - cx) * (-bbox_norm)
            y1_pre = (by1 - cy) * (-bbox_norm)
            x2_pre = (bx2 + cx) * bbox_norm
            y2_pre = (by2 + cy) * bbox_norm

            # Clip to preprocessed image bounds
            x1_pre = max(0.0, min(x1_pre, self.INPUT_WIDTH))
            y1_pre = max(0.0, min(y1_pre, self.INPUT_HEIGHT))
            x2_pre = max(0.0, min(x2_pre, self.INPUT_WIDTH))
            y2_pre = max(0.0, min(y2_pre, self.INPUT_HEIGHT))

            # Ensure x1 <= x2 and y1 <= y2
            if x1_pre > x2_pre:
                x1_pre, x2_pre = x2_pre, x1_pre
            if y1_pre > y2_pre:
                y1_pre, y2_pre = y2_pre, y1_pre

            # Remove letterbox padding and scale back to original image
            x1 = (x1_pre - pad_x) / scale
            y1 = (y1_pre - pad_y) / scale
            x2 = (x2_pre - pad_x) / scale
            y2 = (y2_pre - pad_y) / scale

            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=confidence, class_id=2))

        # Apply NMS
        if len(detections) > 0:
            detections = self._nms(detections)

        return detections

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = [[d.x1, d.y1, d.width, d.height] for d in detections]
        scores = [d.score for d in detections]

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.nms_iou_threshold
        )

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]


class PeopleNetTRT:
    """PeopleNet detector using TensorRT engine."""

    def __init__(
        self,
        engine_path: str,
        conf_threshold: float = 0.5,
        nms_iou_threshold: float = 0.4,
        logger=None,
    ):
        """Initialize TensorRT engine.

        Args:
            engine_path: Path to .engine file
            conf_threshold: Detection confidence threshold
            nms_iou_threshold: NMS IoU threshold
            logger: Optional ROS logger for messages
        """
        import pycuda.autoinit  # noqa: F401 - Required for CUDA context
        import pycuda.driver as cuda
        import tensorrt as trt

        self.logger = logger
        self.engine_path = engine_path
        self.postprocessor = PeopleNetPostProcessor(conf_threshold, nms_iou_threshold)

        # Compute engine hash for provenance tracking
        self.engine_hash = self._compute_hash(engine_path)

        # Load TensorRT engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

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
                self.inputs.append(
                    {"name": name, "host": host_mem, "device": device_mem, "shape": shape}
                )
            else:
                self.outputs.append(
                    {"name": name, "host": host_mem, "device": device_mem, "shape": shape}
                )
                self.output_names.append(name)

        self.stream = cuda.Stream()
        self._gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="peoplenet_gpu")

        if logger:
            logger.info(f"PeopleNet TensorRT engine loaded: {engine_path}")
            logger.info(f"  Engine hash: {self.engine_hash}")
            logger.info(f"  Outputs: {self.output_names}")

    @staticmethod
    def _compute_hash(filepath: str) -> str:
        """Compute first 8 chars of SHA256 for provenance tracking."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:8]

    def _gpu_infer_sync(self):
        """Execute GPU inference pipeline (runs in executor thread)."""
        import pycuda.driver as cuda

        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )
        for inp in self.inputs:
            self.context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], int(out["device"]))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run person detection on BGR image.

        Args:
            image: BGR image (H, W, 3) uint8

        Returns:
            List of Detection objects in original image coordinates
        """
        import pycuda.driver as cuda

        # Preprocess (CPU-side, no timeout needed)
        blob, scale, pad_x, pad_y = self.postprocessor.preprocess(image)

        # Copy input to host buffer (CPU-side)
        np.copyto(self.inputs[0]["host"], blob.ravel())

        t0 = time.perf_counter()
        try:
            future = self._gpu_pool.submit(self._gpu_infer_sync)
            future.result(timeout=PEOPLENET_DETECT_CONTRACT.timeout_sec)
        except TimeoutError:
            timing_ms = round((time.perf_counter() - t0) * 1000, 1)
            _structured_logger.emit_failure(
                operation=PEOPLENET_DETECT_CONTRACT.boundary_name,
                error_code=ErrorCode.TIMEOUT,
                error_detail=f"GPU inference timed out after {PEOPLENET_DETECT_CONTRACT.timeout_sec}s",
                trigger="TimeoutError",
                timing_ms=timing_ms,
                **{k: v for k, v in PEOPLENET_DETECT_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return []
        except Exception as e:
            timing_ms = round((time.perf_counter() - t0) * 1000, 1)
            _structured_logger.emit_failure(
                operation=PEOPLENET_DETECT_CONTRACT.boundary_name,
                error_code=ErrorCode.INTERNAL,
                error_detail=str(e)[:200],
                trigger=type(e).__name__,
                timing_ms=timing_ms,
                **{k: v for k, v in PEOPLENET_DETECT_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return []

        # Find coverage and bbox outputs by name
        coverage = None
        bbox = None

        for out in self.outputs:
            name = out["name"].lower()
            data = out["host"].reshape(out["shape"])
            if "cov" in name or "sigmoid" in name:
                coverage = data
            elif "bbox" in name or "biasadd" in name:
                bbox = data

        if coverage is None or bbox is None:
            if self.logger:
                self.logger.warning(
                    f"Could not identify outputs. Names: {self.output_names}"
                )
            _structured_logger.emit_failure(
                operation="detect_identify_outputs",
                error_code=ErrorCode.INTERNAL,
                error_detail=f"Could not identify outputs. Names: {self.output_names}"[:200],
                trigger="output_tensor_mismatch",
            )
            return []

        return self.postprocessor.postprocess(coverage, bbox, scale, pad_x, pad_y)

    def detect_faces(self, image: np.ndarray, conf_threshold: float = 0.5) -> list[Detection]:
        """Run face detection on BGR image (PeopleNet class 2).

        Args:
            image: BGR image (H, W, 3) uint8
            conf_threshold: Confidence threshold for face detection

        Returns:
            List of Detection objects for faces in original image coordinates
        """
        import pycuda.driver as cuda

        # Preprocess (CPU-side, no timeout needed)
        blob, scale, pad_x, pad_y = self.postprocessor.preprocess(image)

        # Copy input to host buffer (CPU-side)
        np.copyto(self.inputs[0]["host"], blob.ravel())

        t0 = time.perf_counter()
        try:
            future = self._gpu_pool.submit(self._gpu_infer_sync)
            future.result(timeout=PEOPLENET_DETECT_CONTRACT.timeout_sec)
        except TimeoutError:
            timing_ms = round((time.perf_counter() - t0) * 1000, 1)
            _structured_logger.emit_failure(
                operation=PEOPLENET_DETECT_CONTRACT.boundary_name,
                error_code=ErrorCode.TIMEOUT,
                error_detail=f"GPU inference timed out after {PEOPLENET_DETECT_CONTRACT.timeout_sec}s",
                trigger="TimeoutError",
                timing_ms=timing_ms,
                **{k: v for k, v in PEOPLENET_DETECT_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return []
        except Exception as e:
            timing_ms = round((time.perf_counter() - t0) * 1000, 1)
            _structured_logger.emit_failure(
                operation=PEOPLENET_DETECT_CONTRACT.boundary_name,
                error_code=ErrorCode.INTERNAL,
                error_detail=str(e)[:200],
                trigger=type(e).__name__,
                timing_ms=timing_ms,
                **{k: v for k, v in PEOPLENET_DETECT_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return []

        # Find coverage and bbox outputs
        coverage = None
        bbox = None

        for out in self.outputs:
            name = out["name"].lower()
            data = out["host"].reshape(out["shape"])
            if "cov" in name or "sigmoid" in name:
                coverage = data
            elif "bbox" in name or "biasadd" in name:
                bbox = data

        if coverage is None or bbox is None:
            return []

        return self.postprocessor.postprocess_faces(
            coverage, bbox, scale, pad_x, pad_y, conf_threshold
        )

    def detect_timed(
        self, image: np.ndarray
    ) -> tuple[list[Detection], float, float, float]:
        """Run detection with timing breakdown.

        Returns:
            (detections, preprocess_ms, infer_ms, postprocess_ms)
        """
        import pycuda.driver as cuda

        t0 = time.perf_counter()

        # Preprocess
        blob, scale, pad_x, pad_y = self.postprocessor.preprocess(image)
        t1 = time.perf_counter()

        # Copy input to host buffer (CPU-side)
        np.copyto(self.inputs[0]["host"], blob.ravel())

        preprocess_ms = (t1 - t0) * 1000

        t_infer0 = time.perf_counter()
        try:
            future = self._gpu_pool.submit(self._gpu_infer_sync)
            future.result(timeout=PEOPLENET_DETECT_CONTRACT.timeout_sec)
        except TimeoutError:
            timing_ms = round((time.perf_counter() - t_infer0) * 1000, 1)
            _structured_logger.emit_failure(
                operation=PEOPLENET_DETECT_CONTRACT.boundary_name,
                error_code=ErrorCode.TIMEOUT,
                error_detail=f"GPU inference timed out after {PEOPLENET_DETECT_CONTRACT.timeout_sec}s",
                trigger="TimeoutError",
                timing_ms=timing_ms,
                **{k: v for k, v in PEOPLENET_DETECT_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return ([], preprocess_ms, 0.0, 0.0)
        except Exception as e:
            timing_ms = round((time.perf_counter() - t_infer0) * 1000, 1)
            _structured_logger.emit_failure(
                operation=PEOPLENET_DETECT_CONTRACT.boundary_name,
                error_code=ErrorCode.INTERNAL,
                error_detail=str(e)[:200],
                trigger=type(e).__name__,
                timing_ms=timing_ms,
                **{k: v for k, v in PEOPLENET_DETECT_CONTRACT.log_fields().items()
                   if k != "boundary_name"},
            )
            return ([], preprocess_ms, 0.0, 0.0)

        t2 = time.perf_counter()

        # Find outputs
        coverage = None
        bbox = None
        for out in self.outputs:
            name = out["name"].lower()
            data = out["host"].reshape(out["shape"])
            if "cov" in name or "sigmoid" in name:
                coverage = data
            elif "bbox" in name or "biasadd" in name:
                bbox = data

        if coverage is None or bbox is None:
            return [], preprocess_ms, (t2 - t1) * 1000, 0.0

        detections = self.postprocessor.postprocess(coverage, bbox, scale, pad_x, pad_y)
        t3 = time.perf_counter()

        infer_ms = (t2 - t1) * 1000
        postprocess_ms = (t3 - t2) * 1000

        return detections, preprocess_ms, infer_ms, postprocess_ms


def create_detector(
    engine_path: str,
    onnx_path: Optional[str] = None,
    conf_threshold: float = 0.5,
    nms_iou_threshold: float = 0.4,
    logger=None,
) -> PeopleNetTRT:
    """Create PeopleNet detector.

    Unlike face_detection, we do NOT silently fall back to ONNX.
    If engine is missing, fail clearly.

    Args:
        engine_path: Path to TensorRT engine
        onnx_path: Unused (kept for API consistency)
        conf_threshold: Detection confidence threshold
        nms_iou_threshold: NMS IoU threshold
        logger: Optional ROS logger

    Returns:
        PeopleNetTRT detector

    Raises:
        RuntimeError: If engine is missing or incompatible
    """
    if not Path(engine_path).exists():
        msg = (
            f"PeopleNet TensorRT engine not found: {engine_path}\n"
            f"Run build_person_engine.sh to create it."
        )
        if logger:
            logger.fatal(msg)
        _structured_logger.emit_failure(
            operation="create_detector",
            error_code=ErrorCode.NOT_FOUND,
            error_detail=f"Engine not found: {engine_path}"[:200],
            trigger="file_missing",
        )
        raise RuntimeError(msg)

    try:
        return PeopleNetTRT(engine_path, conf_threshold, nms_iou_threshold, logger)
    except Exception as e:
        msg = f"Failed to load PeopleNet engine: {e}\nEngine may be incompatible. Rebuild with build_person_engine.sh"
        if logger:
            logger.fatal(msg)
        _structured_logger.emit_failure(
            operation="create_detector",
            error_code=ErrorCode.INTERNAL,
            error_detail=str(e)[:200],
            trigger=type(e).__name__,
        )
        raise RuntimeError(msg) from e
