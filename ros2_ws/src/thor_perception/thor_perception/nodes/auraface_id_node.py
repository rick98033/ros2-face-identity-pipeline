#!/usr/bin/env python3
"""AuraFace face identification node.

Subscribes to:
    /perception/faces/tracks (thor_msgs/FaceTracks)
    /perception/camera/frame (sensor_msgs/Image)

Publishes:
    /perception/face_id/candidates (thor_msgs/FaceIdentityCandidates)

Architecture:
    - Receives face tracks and camera frames on separate topics
    - Buffers BOTH topics and matches on EITHER arrival (solves frame drop issue)
    - Uses integer nanoseconds for timestamp matching (precision)
    - Crops and aligns faces using 5-point landmarks
    - Computes AuraFace embeddings (TensorRT or ONNX fallback)
    - Matches against enrolled users (max-to-samples cosine similarity)
    - Publishes identity hypotheses with decision states

Decision States:
    - UNKNOWN: score < threshold (no match)
    - AMBIGUOUS: score >= threshold but margin < margin_threshold
    - TENTATIVE: good match, waiting for temporal confirmation
    - CONFIRMED: same identity stable for confirm_time_sec

Key Design Rule:
    Identity hypothesis is always published (top-k + scores).
    Decision state controls whether we treat it as actionable.
    We never confirm AMBIGUOUS.

Buffer-Both Join Strategy:
    Both frames and tracks are buffered keyed by timestamp (integer nanoseconds).
    On either arrival, we check if the matching counterpart exists.
    If yes: process immediately and delete both from buffers.
    If no: leave in buffer for later matching.
    Periodically prune entries older than max_age_sec.
"""

import os
import sys
import time
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from thor_msgs.msg import FaceTracks, FaceIdentityCandidate, FaceIdentityCandidates
from thor_msgs.srv import InferFaceId
from rclpy.callback_groups import ReentrantCallbackGroup
from thor_perception.face_id.alignment import align_face, validate_aligned_crop
from thor_perception.face_id.enrollment_store import FaceEnrollmentStore
import cv2

logger = logging.getLogger(__name__)

from thor_telemetry import get_logger as _get_structured_logger, ErrorCode
from thor_behavior import (
    DegradationPolicy, DegradedState, DepPolicy, FailureMode, RecoveryTarget,
    ResponseType,
)
_structured_logger = _get_structured_logger("auraface_id_node")

_deg_policy = DegradationPolicy("auraface_id", [
    DepPolicy(
        dependency="face_id_model",
        failure_modes=[FailureMode.UNAVAILABLE],
        response=ResponseType.DEGRADE_WITH_NOTIFICATION,
        recovery_target=RecoveryTarget.FULL_OPERATION,
        recovery_action="Check AuraFace model files; verify "
            "model initialization; restart perception container",
    ),
])
_deg_state = DegradedState(_deg_policy, logger=_structured_logger)

# Model paths
DEFAULT_ENGINE_PATH = "/opt/models/face_id/auraface.engine"
DEFAULT_ONNX_PATH = "/opt/models/face_id/auraface.onnx"

# Shadow log path
SHADOW_LOG_PATH = Path.home() / ".thor" / "identity" / "face_id_shadow.jsonl"


def stamp_to_ns(stamp) -> int:
    """Convert ROS stamp to integer nanoseconds for precise matching."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


@dataclass
class TrackState:
    """Per-track embedding buffer and identity state."""

    track_id: int
    embeddings: list[np.ndarray] = field(default_factory=list)
    last_embed_time: float = 0.0
    last_seen_time: float = 0.0  # For TTL-based cleanup
    current_user_id: str = "UNKNOWN"
    current_user_since: float = 0.0  # Monotonic time when identity changed
    confirmed_at: float = 0.0
    # Actual scores for publishing
    last_top_score: float = 0.0
    last_margin: float = 0.0
    alt_user_ids: list[str] = field(default_factory=list)
    alt_scores: list[float] = field(default_factory=list)

    @property
    def centroid(self) -> Optional[np.ndarray]:
        """Compute L2-normalized mean of embeddings."""
        if not self.embeddings:
            return None
        c = np.mean(self.embeddings, axis=0)
        return c / (np.linalg.norm(c) + 1e-8)


class AuraFaceIdNode(Node):
    """AuraFace face identification ROS 2 node."""

    def __init__(self):
        super().__init__("auraface_id_node")

        # Parameters
        self.declare_parameter("enable_face_id", True)
        self.declare_parameter("engine_path", DEFAULT_ENGINE_PATH)
        self.declare_parameter("onnx_path", DEFAULT_ONNX_PATH)
        self.declare_parameter("min_stability", 0.5)
        self.declare_parameter("embed_rate_hz", 2.0)
        self.declare_parameter("buffer_size", 5)
        self.declare_parameter("top_k", 3)
        self.declare_parameter("threshold", 0.5)  # TBD via calibration
        self.declare_parameter("margin_threshold", 0.08)  # Min gap to second-best
        self.declare_parameter("confirm_time_sec", 1.5)
        self.declare_parameter("track_state_ttl_sec", 3.0)  # Keep state after track disappears
        self.declare_parameter("shadow_mode", True)  # Stage 1: log only
        self.declare_parameter("on_demand_mode", False)

        # Buffer-both join parameters
        self.declare_parameter("join_max_age_sec", 5.0)  # Max age before pruning
        self.declare_parameter("join_max_items", 100)  # Max items per buffer

        # Load params
        self.enabled = self.get_parameter("enable_face_id").value
        self._on_demand_mode = self.get_parameter("on_demand_mode").value
        self.engine_path = self.get_parameter("engine_path").value
        self.onnx_path = self.get_parameter("onnx_path").value
        self.min_stability = self.get_parameter("min_stability").value
        self.embed_rate_hz = self.get_parameter("embed_rate_hz").value
        self.buffer_size = self.get_parameter("buffer_size").value
        self.top_k = self.get_parameter("top_k").value
        self.threshold = self.get_parameter("threshold").value
        self.margin_threshold = self.get_parameter("margin_threshold").value
        self.confirm_time_sec = self.get_parameter("confirm_time_sec").value
        self.track_state_ttl_sec = self.get_parameter("track_state_ttl_sec").value
        self.shadow_mode = self.get_parameter("shadow_mode").value
        self.join_max_age_sec = self.get_parameter("join_max_age_sec").value
        self.join_max_items = self.get_parameter("join_max_items").value

        # State
        self.track_states: dict[int, TrackState] = {}
        self.cv_bridge = CvBridge()

        # Buffer-both join: two OrderedDicts keyed by integer nanoseconds
        # OrderedDict maintains insertion order for efficient pruning
        self._frames_by_stamp_ns: OrderedDict[int, np.ndarray] = OrderedDict()
        self._tracks_by_stamp_ns: OrderedDict[int, FaceTracks] = OrderedDict()

        # Diagnostic counters
        self._diag_frames_received = 0
        self._diag_tracks_received = 0
        self._diag_matches_found = 0
        self._diag_frames_expired = 0
        self._diag_tracks_expired = 0

        # Per-gate failure counters (persistent, for metrics)
        self._gate_counters = {
            "processed_total": 0,        # Total faces entering pipeline
            "gated_quality_ok": 0,       # quality_ok=False from face_detection
            "gated_stability": 0,        # stability_score < min_stability
            "gated_rate_limit": 0,       # Rate-limited by embed_rate_hz
            "gated_alignment_fail": 0,   # align_face() returned None
            "gated_black_ratio": 0,      # validate_aligned_crop() failed
            "gated_stale_frame": 0,      # pipeline_age_ms > threshold
            "gated_unknown": 0,          # Catch-all for unclassified gates
            "join_miss_count": 0,        # image-track sync failures
            "embedding_success": 0,      # Reached embedding computation
            "align_fail_count_total": 0, # Total alignment failures (alignment + black_ratio)
        }
        self._gate_counters_lock = threading.Lock()

        # Model state
        self.model = None
        self.is_degraded_mode = False

        # Load enrollment
        self.enrollment = FaceEnrollmentStore()
        user_count = len(self.enrollment.users)
        self.get_logger().info(f"Loaded {user_count} enrolled users")

        # Initialize model
        if self.enabled:
            self._init_model()

        # Callback group for async service
        self._callback_group = ReentrantCallbackGroup()

        # On-demand service (always available)
        self._infer_service = self.create_service(
            InferFaceId,
            "/identity/face/infer",
            self._handle_infer_request,
            callback_group=self._callback_group,
        )

        # Subscribers (only in always-on mode)
        if not self._on_demand_mode:
            # Frame QoS: depth=20 to buffer large messages
            frame_qos = QoSProfile(
                depth=20,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.frame_sub = self.create_subscription(
                Image, "/perception/camera/frame", self._on_frame, frame_qos
            )
            # Tracks QoS
            tracks_qos = QoSProfile(
                depth=20,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            )
            self.tracks_sub = self.create_subscription(
                FaceTracks, "/perception/faces/tracks", self._on_tracks, tracks_qos
            )

            # Publisher (BEST_EFFORT to match perception pipeline pattern)
            candidates_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.candidates_pub = self.create_publisher(
                FaceIdentityCandidates, "/perception/face_id/candidates", candidates_qos
            )

            # Prune timer (1 Hz) to clean up old unmatched entries
            self._prune_timer = self.create_timer(1.0, self._prune_buffers)
        else:
            self.frame_sub = None
            self.tracks_sub = None
            self.candidates_pub = None
            self._prune_timer = None

        # Ensure shadow log directory exists
        if self.shadow_mode:
            SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        mode_str = "on-demand" if self._on_demand_mode else ("SHADOW" if self.shadow_mode else "ACTIVE")
        self.get_logger().info(f"AuraFace ID node started (mode={mode_str}, join=buffer-both)")

    def _handle_infer_request(
        self,
        request: InferFaceId.Request,
        response: InferFaceId.Response,
    ) -> InferFaceId.Response:
        """Handle on-demand face identification service request."""
        start_time = time.monotonic()

        # Check model availability
        if self.model is None:
            response.success = False
            response.error = "no_model"
            response.inference_ms = 0
            return response

        # Check enrollment availability
        if not self.enrollment.users:
            response.success = False
            response.error = "no_enrollment"
            response.inference_ms = 0
            return response

        # Load crop image
        crop_path = Path(request.crop_path)
        if not crop_path.exists():
            response.success = False
            response.error = f"crop_not_found: {request.crop_path}"
            response.inference_ms = 0
            return response

        try:
            # Read and preprocess crop (crop is already 160x160 from crop_buffer)
            crop = cv2.imread(str(crop_path))
            if crop is None:
                response.success = False
                response.error = "crop_read_failed"
                response.inference_ms = 0
                return response

            # Resize to 112x112 for AuraFace
            aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

        except Exception as e:
            response.success = False
            response.error = f"crop_error: {e}"
            response.inference_ms = 0
            return response

        # Compute embedding
        try:
            embedding = self.model.get_embedding(aligned)
            if embedding is None:
                response.success = False
                response.error = "embedding_failed"
                response.inference_ms = int((time.monotonic() - start_time) * 1000)
                return response

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm < 1e-8:
                response.success = False
                response.error = "embedding_zero"
                response.inference_ms = int((time.monotonic() - start_time) * 1000)
                return response
            embedding = embedding / norm

        except Exception as e:
            response.success = False
            response.error = f"inference_error: {e}"
            response.inference_ms = int((time.monotonic() - start_time) * 1000)
            return response

        # Match against enrollment
        matches = self.enrollment.match(embedding, self.top_k)
        top_user, top_score = matches[0] if matches else ("UNKNOWN", 0.0)
        margin = top_score - matches[1][1] if len(matches) > 1 else top_score

        inference_ms = int((time.monotonic() - start_time) * 1000)

        # Populate response
        response.success = True
        response.error = ""
        response.inference_ms = inference_ms

        # Populate candidate
        response.candidate.track_id = request.track_id
        response.candidate.user_id = top_user if top_score >= self.threshold else "UNKNOWN"
        response.candidate.score = top_score
        response.candidate.margin = margin
        response.candidate.embedding_count = 1
        response.candidate.alt_user_ids = [m[0] for m in matches[1:]]
        response.candidate.alt_scores = [m[1] for m in matches[1:]]

        # Determine decision state
        if top_score < self.threshold:
            response.candidate.decision_state = "UNKNOWN"
        elif margin < self.margin_threshold:
            response.candidate.decision_state = "AMBIGUOUS"
        else:
            response.candidate.decision_state = "TENTATIVE"

        self.get_logger().info(
            f"[{request.turn_id}] On-demand face: {response.candidate.user_id} "
            f"(score={top_score:.3f}, latency={inference_ms}ms)"
        )

        return response

    def _init_model(self):
        """Initialize AuraFace TensorRT or ONNX inference."""
        engine_path = Path(self.engine_path)
        onnx_path = Path(self.onnx_path)

        # Try TensorRT first
        if engine_path.exists():
            try:
                self.model = AuraFaceTensorRT(str(engine_path), self.get_logger())
                self.is_degraded_mode = False
                _deg_state.report_recovery("face_id_model")
                self.get_logger().info(f"AuraFace: TensorRT GPU inference (optimal)")
                return
            except Exception as e:
                self.get_logger().error(f"Failed to load TensorRT engine: {e}")
                _structured_logger.emit_failure(
                trace_ctx=None,  # CP-006 §5.1: perception has no turn-level trace
                operation="init_model_tensorrt",
                    error_code=ErrorCode.UNAVAILABLE,
                    error_detail=str(e)[:200],
                    trigger=type(e).__name__,
                )

        # Fall back to ONNX
        if onnx_path.exists():
            try:
                self.model = AuraFaceONNX(str(onnx_path), self.get_logger())
                self.is_degraded_mode = True
                _response = _deg_state.report_degradation(
                    "face_id_model", FailureMode.UNAVAILABLE,
                    "TensorRT engine unavailable — using ONNX Runtime fallback",
                )
                self.get_logger().warn("=" * 60)
                self.get_logger().warn("DEGRADED MODE: Using ONNX Runtime for AuraFace")
                self.get_logger().warn("Run build_auraface_engine.sh for optimal performance")
                self.get_logger().warn("=" * 60)
                return
            except Exception as e:
                self.get_logger().error(f"Failed to load ONNX model: {e}")
                _structured_logger.emit_failure(
                trace_ctx=None,  # CP-006 §5.1: perception has no turn-level trace
                operation="init_model_onnx",
                    error_code=ErrorCode.UNAVAILABLE,
                    error_detail=str(e)[:200],
                    trigger=type(e).__name__,
                )

        # No model available
        self.get_logger().error("=" * 60)
        self.get_logger().error("No AuraFace model found!")
        self.get_logger().error(f"  TensorRT: {engine_path}")
        self.get_logger().error(f"  ONNX: {onnx_path}")
        self.get_logger().error("Run build_auraface_engine.sh to build the model")
        self.get_logger().error("=" * 60)
        _structured_logger.emit_failure(
                trace_ctx=None,  # CP-006 §5.1: perception has no turn-level trace
                operation="init_model",
            error_code=ErrorCode.UNAVAILABLE,
            error_detail="No AuraFace model found (TensorRT and ONNX both unavailable)",
            trigger="model_missing",
        )
        self.model = None
        _response = _deg_state.report_degradation(
            "face_id_model", FailureMode.UNAVAILABLE,
            "Model initialization failed — face ID disabled",
        )

    def _on_frame(self, msg: Image):
        """Buffer frame and attempt match with existing tracks."""
        if not self.enabled:
            return

        try:
            frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            stamp_ns = stamp_to_ns(msg.header.stamp)
            self._diag_frames_received += 1

            # Insert into frame buffer
            self._frames_by_stamp_ns[stamp_ns] = frame

            # Check if matching tracks exist
            if stamp_ns in self._tracks_by_stamp_ns:
                tracks_msg = self._tracks_by_stamp_ns.pop(stamp_ns)
                self._process_matched_pair(frame, tracks_msg)
                # Don't keep the frame since we processed it
                if stamp_ns in self._frames_by_stamp_ns:
                    del self._frames_by_stamp_ns[stamp_ns]

            # Enforce max items (remove oldest)
            while len(self._frames_by_stamp_ns) > self.join_max_items:
                self._frames_by_stamp_ns.popitem(last=False)
                self._diag_frames_expired += 1

            # Diagnostic: log every 50 frames
            if self._diag_frames_received % 50 == 1:
                stamp_sec = stamp_ns / 1e9
                self.get_logger().info(
                    f"DIAG_FRAME: recv={self._diag_frames_received} stamp={stamp_sec:.3f} "
                    f"frame_buf={len(self._frames_by_stamp_ns)} track_buf={len(self._tracks_by_stamp_ns)} "
                    f"matches={self._diag_matches_found} expired_f={self._diag_frames_expired} expired_t={self._diag_tracks_expired}"
                )

        except Exception as e:
            self.get_logger().warning(f"Frame decode failed: {e}")
            _structured_logger.emit_failure(
                trace_ctx=None,  # CP-006 §5.1: perception has no turn-level trace
                operation="frame_decode",
                error_code=ErrorCode.INTERNAL,
                error_detail=str(e)[:200],
                trigger=type(e).__name__,
            )

    def _on_tracks(self, msg: FaceTracks):
        """Buffer tracks and attempt match with existing frame."""
        if not self.enabled:
            return

        stamp_ns = stamp_to_ns(msg.header.stamp)
        self._diag_tracks_received += 1

        # Check if matching frame exists
        if stamp_ns in self._frames_by_stamp_ns:
            frame = self._frames_by_stamp_ns.pop(stamp_ns)
            self._process_matched_pair(frame, msg)
            # Don't buffer tracks since we processed them
        else:
            # No frame yet - buffer tracks for later
            self._tracks_by_stamp_ns[stamp_ns] = msg

            # Enforce max items (remove oldest)
            while len(self._tracks_by_stamp_ns) > self.join_max_items:
                self._tracks_by_stamp_ns.popitem(last=False)
                self._diag_tracks_expired += 1

        # Diagnostic: log every 50 tracks
        if self._diag_tracks_received % 50 == 1:
            stamp_sec = stamp_ns / 1e9
            hit_rate = self._diag_matches_found / max(1, self._diag_tracks_received)
            with self._gate_counters_lock:
                join_misses = self._gate_counters["join_miss_count"]
            self.get_logger().info(
                f"DIAG_TRACKS: recv={self._diag_tracks_received} stamp={stamp_sec:.3f} "
                f"frame_buf={len(self._frames_by_stamp_ns)} track_buf={len(self._tracks_by_stamp_ns)} "
                f"hit_rate={hit_rate:.1%} join_misses={join_misses}"
            )

    def get_gate_counters(self) -> dict:
        """Get current gate counter values (thread-safe).

        Returns a copy of all gate counters for diagnostics.
        Invariant: gated_* counters + embedding_success = processed_total
        """
        with self._gate_counters_lock:
            return dict(self._gate_counters)

    def _prune_buffers(self):
        """Prune old entries from both buffers (called by timer)."""
        now_ns = time.time_ns()
        max_age_ns = int(self.join_max_age_sec * 1e9)
        cutoff_ns = now_ns - max_age_ns

        # Prune old frames
        old_frame_stamps = [s for s in self._frames_by_stamp_ns if s < cutoff_ns]
        for stamp_ns in old_frame_stamps:
            del self._frames_by_stamp_ns[stamp_ns]
            self._diag_frames_expired += 1

        # Prune old tracks - these are actual join misses (tracks that never matched a frame)
        old_track_stamps = [s for s in self._tracks_by_stamp_ns if s < cutoff_ns]
        for stamp_ns in old_track_stamps:
            del self._tracks_by_stamp_ns[stamp_ns]
            self._diag_tracks_expired += 1

        # Count actual join misses (tracks pruned without matching)
        if old_track_stamps:
            with self._gate_counters_lock:
                self._gate_counters["join_miss_count"] += len(old_track_stamps)

        # Log if significant pruning occurred
        if old_frame_stamps or old_track_stamps:
            self.get_logger().debug(
                f"PRUNE: frames={len(old_frame_stamps)} tracks={len(old_track_stamps)} (join_miss) "
                f"remaining_f={len(self._frames_by_stamp_ns)} remaining_t={len(self._tracks_by_stamp_ns)}"
            )

    def _process_matched_pair(self, frame: np.ndarray, msg: FaceTracks):
        """Process a matched frame+tracks pair."""
        self._diag_matches_found += 1
        current_time = time.monotonic()
        embed_interval = 1.0 / self.embed_rate_hz

        candidates = []
        tracks_processed = 0
        total_latency = 0.0
        active_track_ids = set()

        # Local gating counters (for this batch)
        batch_gated_quality = 0
        batch_gated_stability = 0
        batch_gated_rate_limit = 0
        batch_gated_alignment = 0
        batch_gated_black_ratio = 0
        batch_embedding_success = 0

        for track in msg.tracks:
            active_track_ids.add(track.track_id)

            # Increment processed total
            with self._gate_counters_lock:
                self._gate_counters["processed_total"] += 1

            # Gate by quality
            if not track.quality_ok:
                batch_gated_quality += 1
                with self._gate_counters_lock:
                    self._gate_counters["gated_quality_ok"] += 1
                continue

            # Gate by stability
            if track.stability_score < self.min_stability:
                batch_gated_stability += 1
                with self._gate_counters_lock:
                    self._gate_counters["gated_stability"] += 1
                continue

            # Get or create track state
            if track.track_id not in self.track_states:
                self.track_states[track.track_id] = TrackState(
                    track_id=track.track_id, current_user_since=current_time
                )
            state = self.track_states[track.track_id]
            state.last_seen_time = current_time

            # Rate-limit embedding computation
            if current_time - state.last_embed_time < embed_interval:
                batch_gated_rate_limit += 1
                with self._gate_counters_lock:
                    self._gate_counters["gated_rate_limit"] += 1
                candidates.append(self._make_candidate(track, state, is_ambiguous=False))
                continue

            # Compute embedding (with detailed failure tracking)
            start = time.monotonic()
            embedding, gate_reason = self._compute_embedding_with_reason(track, frame)
            latency = (time.monotonic() - start) * 1000
            total_latency += latency

            if embedding is None:
                # Track specific gate reason
                with self._gate_counters_lock:
                    if gate_reason == "alignment_fail":
                        batch_gated_alignment += 1
                        self._gate_counters["gated_alignment_fail"] += 1
                        self._gate_counters["align_fail_count_total"] += 1
                    elif gate_reason == "black_ratio":
                        batch_gated_black_ratio += 1
                        self._gate_counters["gated_black_ratio"] += 1
                        self._gate_counters["align_fail_count_total"] += 1
                    else:
                        self._gate_counters["gated_unknown"] += 1
                continue

            # Successful embedding
            batch_embedding_success += 1
            with self._gate_counters_lock:
                self._gate_counters["embedding_success"] += 1

            # Update buffer (rolling)
            state.embeddings.append(embedding)
            if len(state.embeddings) > self.buffer_size:
                state.embeddings.pop(0)
            state.last_embed_time = current_time

            # Match against enrollment
            centroid = state.centroid
            if centroid is None:
                candidates.append(self._make_candidate(track, state, is_ambiguous=False))
                continue

            matches = self.enrollment.match(centroid, self.top_k)
            top_user, top_score = matches[0]
            margin = top_score - matches[1][1] if len(matches) > 1 else top_score

            # Store actual scores for publishing
            state.last_top_score = top_score
            state.last_margin = margin
            state.alt_user_ids = [m[0] for m in matches[1:]]
            state.alt_scores = [m[1] for m in matches[1:]]

            # Identity hypothesis: always keep top candidate
            # Decision state controls whether it's actionable
            if top_score < self.threshold:
                new_user_id = "UNKNOWN"
                is_ambiguous = False
            elif margin < self.margin_threshold:
                new_user_id = top_user  # Keep hypothesis, but mark AMBIGUOUS
                is_ambiguous = True
            else:
                new_user_id = top_user
                is_ambiguous = False

            # Track identity stability using monotonic time
            if new_user_id != state.current_user_id:
                state.current_user_id = new_user_id
                state.current_user_since = current_time
                state.confirmed_at = 0.0

            # Check confirmation (never confirm AMBIGUOUS)
            time_at_current = current_time - state.current_user_since
            if (
                not is_ambiguous
                and time_at_current >= self.confirm_time_sec
                and state.confirmed_at == 0.0
            ):
                state.confirmed_at = current_time

            tracks_processed += 1
            candidate = self._make_candidate(track, state, is_ambiguous)
            candidates.append(candidate)

            # Shadow logging
            if self.shadow_mode:
                self._log_shadow(track, state, is_ambiguous)

        # Cleanup stale track states (with TTL, not immediate)
        stale_ids = []
        for tid, st in self.track_states.items():
            if tid not in active_track_ids:
                if current_time - st.last_seen_time > self.track_state_ttl_sec:
                    stale_ids.append(tid)
        for stale_id in stale_ids:
            del self.track_states[stale_id]

        # Publish
        out_msg = FaceIdentityCandidates()
        out_msg.header = msg.header
        out_msg.candidates = candidates
        out_msg.tracks_processed = tracks_processed
        out_msg.tracks_gated_out = batch_gated_quality + batch_gated_stability + batch_gated_alignment
        out_msg.avg_embed_latency_ms = total_latency / max(1, tracks_processed)

        self.candidates_pub.publish(out_msg)

        # Detailed gate logging when faces exist but no candidates
        total_tracks = len(msg.tracks)
        if total_tracks > 0 and len(candidates) == 0:
            self.get_logger().warn(
                f"GATE_DEBUG: tracks={total_tracks} candidates=0 "
                f"quality={batch_gated_quality} stability={batch_gated_stability} "
                f"rate_limit={batch_gated_rate_limit} alignment={batch_gated_alignment} "
                f"black_ratio={batch_gated_black_ratio}"
            )

    def _compute_embedding(self, track, frame: np.ndarray) -> Optional[np.ndarray]:
        """Crop, align, and compute embedding for a track."""
        embedding, _ = self._compute_embedding_with_reason(track, frame)
        return embedding

    def _compute_embedding_with_reason(self, track, frame: np.ndarray) -> tuple[Optional[np.ndarray], str]:
        """Crop, align, and compute embedding for a track with failure reason.

        Returns:
            (embedding, gate_reason) where gate_reason is one of:
            - "no_model": Model not loaded
            - "alignment_fail": Affine transform estimation failed
            - "black_ratio": Aligned crop has too much black (bad alignment)
            - "inference_fail": Model inference failed
            - "zero_norm": Embedding has zero norm
            - "": Success
        """
        if self.model is None:
            return None, "no_model"

        # Extract landmarks using explicit fields (FaceLandmarks.msg)
        # NOTE: Landmarks are in SOURCE FRAME coordinates (same resolution as 'frame')
        # Convention: subject-left (left_eye appears on VIEWER'S RIGHT for frontal face)
        landmarks = np.array(
            [
                track.landmarks.left_eye_x,
                track.landmarks.left_eye_y,
                track.landmarks.right_eye_x,
                track.landmarks.right_eye_y,
                track.landmarks.nose_tip_x,
                track.landmarks.nose_tip_y,
                track.landmarks.left_mouth_x,
                track.landmarks.left_mouth_y,
                track.landmarks.right_mouth_x,
                track.landmarks.right_mouth_y,
            ],
            dtype=np.float32,
        )

        # Align face (returns None if alignment fails)
        # Debug: manually check alignment steps
        lm_reshaped = landmarks.reshape(5, 2)
        # Subject-left to viewer-perspective swap
        lm_swapped = lm_reshaped[[1, 0, 2, 4, 3], :]
        from thor_perception.face_id.alignment import ARCFACE_DST
        M, inliers = cv2.estimateAffinePartial2D(lm_swapped, ARCFACE_DST)
        inlier_count = inliers.sum() if inliers is not None else 0

        aligned = align_face(frame, landmarks)
        if aligned is None:
            self.get_logger().warn(
                f"ALIGN_FAIL: track={track.track_id} M={M is not None} inliers={inlier_count} "
                f"lm_swapped={lm_swapped.flatten()[:6]}..."
            )
            return None, "alignment_fail"

        # Safety check: reject crops with too much black (bad alignment)
        if not validate_aligned_crop(aligned, max_black_ratio=0.3):
            black_ratio = np.mean(aligned < 10) if aligned is not None else 1.0
            self.get_logger().warn(f"ALIGN_FAIL: track={track.track_id} black_ratio={black_ratio:.2f} > 0.3")
            return None, "black_ratio"

        # Run model inference
        embedding = self.model.get_embedding(aligned)
        if embedding is None:
            return None, "inference_fail"

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return None, "zero_norm"
        return embedding / norm, ""

    def _make_candidate(
        self, track, state: TrackState, is_ambiguous: bool
    ) -> FaceIdentityCandidate:
        """Build candidate message from track state.

        Identity hypothesis is always published (top-k + scores).
        Decision state controls whether we treat it as actionable.
        We never confirm AMBIGUOUS.
        """
        msg = FaceIdentityCandidate()
        msg.track_id = track.track_id
        msg.user_id = state.current_user_id
        msg.score = state.last_top_score
        msg.margin = state.last_margin
        msg.embedding_count = len(state.embeddings)
        msg.alt_user_ids = state.alt_user_ids
        msg.alt_scores = state.alt_scores

        if state.current_user_id == "UNKNOWN":
            msg.decision_state = "UNKNOWN"
        elif is_ambiguous:
            msg.decision_state = "AMBIGUOUS"
        elif state.confirmed_at > 0:
            msg.decision_state = "CONFIRMED"
        else:
            msg.decision_state = "TENTATIVE"

        return msg

    def _log_shadow(self, track, state: TrackState, is_ambiguous: bool):
        """Log identity decision to shadow log for calibration."""
        try:
            decision = "AMBIGUOUS" if is_ambiguous else (
                "UNKNOWN" if state.current_user_id == "UNKNOWN" else
                "CONFIRMED" if state.confirmed_at > 0 else "TENTATIVE"
            )
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "track_id": track.track_id,
                "user_id": state.current_user_id,
                "score": round(state.last_top_score, 4),
                "margin": round(state.last_margin, 4),
                "decision": decision,
                "stability_score": round(track.stability_score, 3),
                "embedding_count": len(state.embeddings),
            }
            with open(SHADOW_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.get_logger().debug(f"Shadow log failed: {e}")


# =============================================================================
# AuraFace Inference Backends
# =============================================================================


class AuraFaceTensorRT:
    """AuraFace embedding using TensorRT engine."""

    def __init__(self, engine_path: str, logger=None):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 - Required for CUDA context

        self.logger = logger

        # Load TensorRT engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.output_shapes = {}

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
                self.output_shapes[name] = shape

        self.stream = cuda.Stream()

        if logger:
            logger.info(f"AuraFace TensorRT engine loaded: {engine_path}")

    def get_embedding(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """Compute face embedding from aligned 112x112 BGR image."""
        import pycuda.driver as cuda

        # Preprocess: BGR 112x112 -> CHW float32
        # AuraFace expects BGR, no normalization needed (model handles it)
        blob = aligned_face.astype(np.float32)
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # Add batch dim
        blob = np.ascontiguousarray(blob)

        # Copy input to device
        np.copyto(self.inputs[0]["host"], blob.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        # Set tensor addresses
        for inp in self.inputs:
            self.context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], int(out["device"]))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs back
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        # Return first output (should be 512-dim embedding)
        if self.outputs:
            embedding = self.outputs[0]["host"].copy()
            return embedding.flatten()
        return None


class AuraFaceONNX:
    """AuraFace embedding using ONNX Runtime (CPU fallback)."""

    def __init__(self, model_path: str, logger=None):
        import onnxruntime as ort

        self.logger = logger

        # Force CPU provider for fallback mode
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        if logger:
            logger.info(f"AuraFace ONNX model loaded: {model_path}")

    def get_embedding(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """Compute face embedding from aligned 112x112 BGR image."""
        # Preprocess: BGR 112x112 -> CHW float32
        blob = aligned_face.astype(np.float32)
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # Add batch dim

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        embedding = outputs[0].flatten()
        return embedding


# =============================================================================
# Entry Point
# =============================================================================


def main(args=None):
    rclpy.init(args=args)
    node = AuraFaceIdNode()

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
